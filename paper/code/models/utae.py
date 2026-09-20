import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Building Blocks
# ============================================================================

class TemporalAggregator(nn.Module):
    def __init__(self, mode="att_group"):
        super().__init__()
        self.mode = mode

    def forward(self, x, pad_mask, attn_mask):
        if self.mode == "att_group":
            n_heads, B, T, h_attn, w_attn = attn_mask.shape
            _, _, C, H, W = x.shape
            hd = C // n_heads
            chunks = []
            for h in range(n_heads):
                a = attn_mask[h]
                if H != h_attn or W != w_attn:
                    a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                if pad_mask is not None and pad_mask.any():
                    a = a * (~pad_mask).float()[:, :, None, None]
                chunks.append((a.unsqueeze(2) * x[:, :, h*hd:(h+1)*hd]).sum(dim=1))
            return torch.cat(chunks, dim=1)
        else:
            if pad_mask is not None and pad_mask.any():
                m = (~pad_mask).float()[:, :, None, None, None]
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(dim=1)


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None):
        super().__init__()
        self.d, self.repeat = d, repeat
        self.register_buffer('denom', torch.pow(T, 2 * (torch.arange(d).float() // 2) / d))

    def forward(self, positions):
        s = positions[:, :, None] / self.denom[None, None, :]
        s[:, :, 0::2] = torch.sin(s[:, :, 0::2])
        s[:, :, 1::2] = torch.cos(s[:, :, 1::2])
        return s.repeat(1, 1, self.repeat) if self.repeat else s


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), float('-inf'))
        attn = self.dropout(self.softmax(attn).nan_to_num(0.0))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head, self.d_k, self.d_in = n_head, d_k, d_in
        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
        nn.init.normal_(self.Q, mean=0, std=(2.0 / d_k) ** 0.5)
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=(2.0 / d_k) ** 0.5)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, v, pad_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        B, T, _ = v.size()
        q = self.Q.unsqueeze(1).expand(-1, B, -1).reshape(-1, d_k)
        k = self.fc1_k(v).view(B, T, n_head, d_k).permute(2, 0, 1, 3).reshape(-1, T, d_k)
        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * B, T, -1)
        output, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(n_head, B, 1, T).squeeze(2)
        output = output.view(n_head, B, 1, d_in // n_head).squeeze(2)
        return output, attn


class LTAE2d(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=4, mlp=[256, 128],
                 dropout=0.2, d_model=256, T=1000, return_att=False, positional_encoding=True):
        super().__init__()
        self.in_channels = in_channels
        self.return_att = return_att
        self.n_head = n_head
        self.d_model = d_model if d_model is not None else in_channels
        self.inconv = nn.Conv1d(in_channels, d_model, 1) if d_model is not None else None
        self.positional_encoder = PositionalEncoder(
            self.d_model // n_head, T=T, repeat=n_head) if positional_encoding else None
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])
        layers = []
        mlp_dims = [self.d_model] + mlp[1:] if mlp[0] == self.d_model else mlp
        for i in range(len(mlp_dims) - 1):
            layers.extend([nn.Linear(mlp_dims[i], mlp_dims[i+1]),
                           nn.BatchNorm1d(mlp_dims[i+1]), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        B, T, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = pad_mask[:, :, None, None].expand(-1, -1, h, w).permute(0, 2, 3, 1).reshape(B*h*w, T)
        out = x.permute(0, 3, 4, 1, 2).reshape(B*h*w, T, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.positional_encoder is not None and batch_positions is not None:
            # A single (B, T) tensor, or a tuple of several -- one sinusoidal
            # encoding per field, summed. Lets a mode like "relative_doy" combine
            # two references without the encoder needing extra parameters (it has
            # none: PositionalEncoder is a fixed buffer, so summing is free).
            bps = batch_positions if isinstance(batch_positions, tuple) else (batch_positions,)
            for bp in bps:
                bp = bp[:, :, None, None].expand(-1, -1, h, w).permute(0, 2, 3, 1).reshape(B*h*w, T)
                out = out + self.positional_encoder(bp)
        out, attn = self.attention_heads(out, pad_mask=pad_mask)
        out = out.permute(1, 0, 2).reshape(B*h*w, -1)
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out).view(B, h, w, -1).permute(0, 3, 1, 2)
        attn = attn.view(self.n_head, B, h, w, T).permute(0, 1, 4, 2, 3)
        return (out, attn) if self.return_att else out


class TemporallySharedBlock(nn.Module):
    def __init__(self, pad_value=None):
        super().__init__()
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        b, t, c, h, w = input.shape
        out = self.forward(input.reshape(b * t, c, h, w))
        return out.view(b, t, *out.shape[1:])


class ConvLayer(nn.Module):
    def __init__(self, nkernels, norm="batch", k=3, s=1, p=1, n_groups=4,
                 last_relu=True, padding_mode="reflect"):
        super().__init__()
        nl = {"batch": nn.BatchNorm2d, "instance": nn.InstanceNorm2d,
              "group": lambda c: nn.GroupNorm(n_groups, c)}.get(norm)
        layers = []
        for i in range(len(nkernels) - 1):
            layers.append(nn.Conv2d(nkernels[i], nkernels[i+1], k, padding=p, stride=s,
                                    padding_mode=padding_mode))
            if nl: layers.append(nl(nkernels[i+1]))
            if last_relu or i < len(nkernels) - 2: layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(TemporallySharedBlock):
    def __init__(self, nkernels, pad_value=None, norm="batch", last_relu=True, padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(nkernels, norm=norm, last_relu=last_relu, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    def __init__(self, d_in, d_out, k, s, p, pad_value=None, norm="batch", padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer([d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode)
        self.conv1 = ConvLayer([d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(self.down(x))
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(nn.Conv2d(d, d, 1), nn.BatchNorm2d(d), nn.ReLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(d_in, d_out, k, stride=s, padding=p),
                                nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv1 = ConvLayer([d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x, skip):
        out = torch.cat([self.up(x), self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)



# ============================================================================
# UTAE_MM — Multi-Modal U-TAE
# ============================================================================
#
# All strategies share the same U-TAE backbone (spatial CNN encoder,
# L-TAE temporal attention, U-Net decoder with attention-weighted skips).
# They differ only in WHERE the modality streams merge and HOW the decoder
# skip connections are formed.
#
# Positional encoding uses real acquisition days, not timestep indices, so
# irregular sampling is handled natively by every strategy.
#
# ----------------------------------------------------------------------------
#
#   "single"   — Baseline, one SITS modality only.
#                1 spatial encoder, 1 LTAE, 1 decoder.
#                Skips: attention-weighted temporal pooling of encoder feats.
#
#   "early"    — Merge at the input, channel-wise.
#                Non-reference modalities are linearly resampled onto the
#                reference timeline (S2 if present) before concatenation,
#                so unaligned acquisition dates are handled by interpolation
#                rather than by requiring aligned inputs.
#                1 spatial encoder over C_total channels, 1 LTAE, 1 decoder.
#                Cheapest in parameters. Modality identity is lost immediately,
#                and a corrupted modality contaminates every feature map.
#
#   "mid"      — Merge at the sequence level, after spatial encoding.
#                Per-modality spatial encoders (separate weights) produce
#                bottleneck features. These are concatenated along T into one
#                combined sequence sorted chronologically by acquisition day,
#                then passed through 1 shared LTAE and 1 decoder.
#                No temporal alignment or resampling needed. The LTAE attends
#                across sensors and time jointly, so a single attention map
#                weights all acquisitions regardless of source.
#                Skips: per-modality features concatenated along T, sorted,
#                then pooled with the shared attention map.
#
#   "late"     — Merge at the bottleneck, channel-wise.
#                Per-modality spatial encoders AND per-modality LTAEs, so each
#                stream keeps its own temporal attention. Pooled embeddings are
#                concatenated along C and compressed by a 1x1 fusion conv into
#                a single decoder.
#                Skips: per-modality pooled skips concatenated along C, which
#                widens every skip by a factor of M and grows decoder cost.
#
#   "decision" — Merge at the output, logit-wise.
#                Fully separate pipelines per modality (spatial encoder, LTAE,
#                bottleneck, decoder, output head). Logits are combined by a
#                softmax over M learned scalar weights, one global weight per
#                modality with no spatial or channel dependence.
#                Most expensive in parameters. No cross-modal interaction at
#                any point before the final sum.
#
# ----------------------------------------------------------------------------
# DEM (non-temporal, optional):
#   Small CNN produces multi-scale features that are concatenated to the
#   decoder skips at every level, for all strategies. No temporal encoding,
#   no attention.
#
# ============================================================================

# The temporal references the paper reports, identical for UTAE_MM and TSViT so
# that "TSViT vs U-TAE" compares the architectures and not two notions of time.
# Each maps to the metadata field(s) that feed the positional encoder -- more
# than one field means their sinusoidal encodings are SUMMED (PositionalEncoder
# has no learnable parameters, so this is free and keeps the encoder dimension
# fixed regardless of mode). All fields are derived from the SELECTED
# acquisition timestamps and none carries the event date or the absolute
# calendar date.
#
#   relative      Days since the sample's first frame, p_t = d_t - min_m d_0^(m).
#                 The default. Monotone, bounded by the window length,
#                 identically distributed over train/val/test. Carries the
#                 spacing between acquisitions and nothing else -- no
#                 seasonality, since day 0 is a different calendar date in
#                 every sample.
#   doy           Day of year (1..366), the reference U-TAE and TSViT use in
#                 their published form. Comparable across the whole dataset --
#                 day 100 always means "~April", whichever sample or year --
#                 which is what actually carries seasonality. But it is cyclic:
#                 88 % of the flood series and 66 % of the fire series cross a
#                 year boundary, where it decreases over the sequence.
#   relative_doy  Both, summed: correct order and spacing from `relative`,
#                 seasonal signal from `doy`. The cyclic jump in `doy` is
#                 harmless here because `relative` alone already disambiguates
#                 which frame is which -- `doy` only has to describe season, not
#                 carry order.
#
# `relative` doubles as the axis fusion uses to align and sort S1/S2 (see
# `forward`) -- that role is unconditional, independent of `temporal_pe`.
# Swapping it for `doy` there would sort by season instead of chronology and
# corrupt any sample crossing a year boundary. See data/utils.relative_days and
# data/utils.day_of_year for the measurements behind these sentences.
TEMPORAL_PE_MODES = {
    "relative":     ("rel_day",),
    "doy":          ("doy",),
    "relative_doy": ("rel_day", "doy"),
}


class UTAE_MM(nn.Module):

    def __init__(self, modality_infos, num_classes, fusion_stage="mid",
                 encoder_widths=[64, 64, 64, 128], decoder_widths=[128, 64, 32, 32],
                 dem_enc_width=32, str_conv_k=4, str_conv_s=2, str_conv_p=1,
                 agg_mode="att_group", encoder_norm="group",
                 n_head=16, d_model=256, d_k=4,
                 padding_mode="reflect", temporal_pe="relative"):
        super().__init__()

        assert fusion_stage in ("single", "early", "mid", "late", "decision")
        assert temporal_pe in TEMPORAL_PE_MODES, (
            f"temporal_pe must be one of {sorted(TEMPORAL_PE_MODES)}, got {temporal_pe!r}")
        self.temporal_pe = temporal_pe
        self.fusion_stage = fusion_stage
        self.n_stages = len(encoder_widths)
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.num_classes = num_classes
        self.agg_mode = agg_mode

        # ── Parse modalities ──
        self.sits_keys = [m for m, info in modality_infos.items() if info.get("temporal", True)]
        s2_candidates = [m for m in self.sits_keys if m.upper().startswith("S2")]
        self.early_ref_mod = s2_candidates[0] if s2_candidates else self.sits_keys[0]
        self.has_dem = ("DEM" in modality_infos and not modality_infos["DEM"].get("temporal", True))
        self.dem_enc_width = dem_enc_width if self.has_dem else 0
        n_mods = len(self.sits_keys)

        if fusion_stage == "single":
            assert n_mods == 1, f"fusion_stage=single requires exactly one SITS modality, got {n_mods}"

        # ── Encoder groups (S1ASC + S1DSC)──
        mod_ch = {m: modality_infos[m]["channels"] for m in self.sits_keys}

        # ── Shared builder kwargs ──
        ck = dict(norm=encoder_norm, padding_mode=padding_mode)
        dk = dict(k=str_conv_k, s=str_conv_s, p=str_conv_p, norm=encoder_norm, padding_mode=padding_mode)
        uk = dict(k=str_conv_k, s=str_conv_s, p=str_conv_p, norm=encoder_norm, padding_mode=padding_mode)
        ltae_kw = dict(d_model=d_model, n_head=n_head, mlp=[d_model, encoder_widths[-1]], return_att=True, d_k=d_k)

        # ════════════════════════════════════════════════════════════════
        # SPATIAL ENCODERS
        # ════════════════════════════════════════════════════════════════

        if fusion_stage == "early":
            total_ch = sum(modality_infos[m]["channels"] for m in self.sits_keys)
            self.in_conv = ConvBlock([total_ch, encoder_widths[0], encoder_widths[0]], **ck)
            self.down_blocks = nn.ModuleList([DownConvBlock(encoder_widths[i], encoder_widths[i+1], **dk) for i in range(self.n_stages - 1)])
        else:
            self.in_convs = nn.ModuleDict({m: ConvBlock([mod_ch[m], encoder_widths[0], encoder_widths[0]], **ck) for m in self.sits_keys})
            self.down_blocks = nn.ModuleDict({
                m: nn.ModuleList([
                    DownConvBlock(encoder_widths[i], encoder_widths[i+1], **dk)
                    for i in range(self.n_stages - 1)])
                for m in self.sits_keys})

        # ════════════════════════════════════════════════════════════════
        # TEMPORAL ENCODERS
        # ════════════════════════════════════════════════════════════════

        if fusion_stage in ("single", "early", "mid"):
            # Single shared LTAE
            self.temporal_encoder = LTAE2d(encoder_widths[-1], **ltae_kw)
        else:
            # Per-modality LTAEs (late, decision)
            self.temporal_encoders = nn.ModuleDict({m: LTAE2d(encoder_widths[-1], **ltae_kw) for m in self.sits_keys})

        self.temporal_aggregator = TemporalAggregator(mode=agg_mode)

        # ════════════════════════════════════════════════════════════════
        # DECODERS
        # ════════════════════════════════════════════════════════════════

        if fusion_stage in ("single", "early", "mid"):
            # Single decoder path
            self.bottleneck = nn.Sequential(nn.Conv2d(encoder_widths[-1], decoder_widths[0], 1), nn.BatchNorm2d(decoder_widths[0]), nn.ReLU(inplace=True))
            self.up_blocks = nn.ModuleList([
                UpConvBlock(decoder_widths[i], decoder_widths[i+1], d_skip=encoder_widths[-(i+2)] + self.dem_enc_width, **uk)
                for i in range(self.n_stages - 1)])
            self.out_conv = ConvBlock([decoder_widths[-1], 32, num_classes],padding_mode=padding_mode, last_relu=False)

        elif fusion_stage in ("late"):
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(encoder_widths[-1] * n_mods, decoder_widths[0], 1),
                nn.BatchNorm2d(decoder_widths[0]), nn.ReLU(inplace=True))

            # Skip width: "late" concats the M modality skips along channels (* n_mods).
            self.up_blocks = nn.ModuleList([
                UpConvBlock(decoder_widths[i], decoder_widths[i+1], d_skip=encoder_widths[-(i+2)] * n_mods + self.dem_enc_width, **uk)
                for i in range(self.n_stages - 1)])
            self.out_conv = ConvBlock(
                [decoder_widths[-1], 32, num_classes],
                padding_mode=padding_mode, last_relu=False)

        else:  # decision
            self.bottlenecks = nn.ModuleDict({
                m: nn.Sequential(
                    nn.Conv2d(encoder_widths[-1], decoder_widths[0], 1),
                    nn.BatchNorm2d(decoder_widths[0]), nn.ReLU(inplace=True))
                for m in self.sits_keys})
            self.up_blocks_dict = nn.ModuleDict({
                m: nn.ModuleList([
                    UpConvBlock(decoder_widths[i], decoder_widths[i+1],
                                d_skip=encoder_widths[-(i+2)] + self.dem_enc_width, **uk)
                    for i in range(self.n_stages - 1)])
                for m in self.sits_keys})
            self.out_convs = nn.ModuleDict({
                m: ConvBlock([decoder_widths[-1], 32, num_classes],
                             padding_mode=padding_mode, last_relu=False)
                for m in self.sits_keys})
            self.decision_weights = nn.Parameter(torch.ones(n_mods))

        # ════════════════════════════════════════════════════════════════
        # DEM ENCODER  (naive: CNN → multi-scale → concat at skip)
        # ════════════════════════════════════════════════════════════════
        if self.has_dem:
            self.dem_in_conv = ConvBlock(
                [modality_infos["DEM"]["channels"], dem_enc_width, dem_enc_width], **ck)
            self.dem_down_blocks = nn.ModuleList([
                DownConvBlock(dem_enc_width, dem_enc_width, **dk)
                for _ in range(self.n_stages - 2)])

        for p in self.parameters():
            p.data = p.data.contiguous()

        print(f"[UTAE-MM] fusion={fusion_stage} | SITS={self.sits_keys} | "
              f"DEM={'yes' if self.has_dem else 'no'} | "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _encode_dem(self, image: dict) -> list[torch.Tensor] | None:
        if not self.has_dem or "DEM" not in image:
            return None
        dem = image["DEM"]
        if dem.dim() == 5:           # (B, 1, C, H, W) -> (B, C, H, W)
            dem = dem.squeeze(1)
        feats = [self.dem_in_conv(dem)]
        for down in self.dem_down_blocks:
            feats.append(down(feats[-1]))
        return feats

    def _cat_skip(self, skip_parts, dem_feats, level):
        if dem_feats is not None:
            skip_parts.append(dem_feats[level])
        return torch.cat(skip_parts, dim=1)

    def _encode_sits_per_mod(self, image: dict) -> dict:
        all_feats = {}
        for m in self.sits_keys:
            feats = [self.in_convs[m].smart_forward(image[m])]
            for down in self.down_blocks[m]:
                feats.append(down.smart_forward(feats[-1]))
            all_feats[m] = feats
        return all_feats

    def _temporal_pool_per_mod(self, all_feats: dict, pe: dict) -> tuple[dict, dict]:
        """Per-modality LTAE pooling. `pe` is the dict TEMPORAL_PE_MODES
        selected, each value a per-modality (B, T_m) positions tensor."""
        pooled, attns = {}, {}
        for m in self.sits_keys:
            out, attn = self.temporal_encoders[m](
                all_feats[m][-1],
                batch_positions=self._pe_positions(pe, m),
                pad_mask=None, # all timesteps should be valid
            )
            pooled[m], attns[m] = out, attn
        return pooled, attns
    
    def _build_sorted_timeline(self, days: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Merge per-modality `days` into one chronologically sorted timeline.

        Returns
        -------
        sorted_dates : (B, T_total) or None
        sort_idx     : (B, T_total) — gather indices to reorder concatenated tensors
        """
        dates_list = [days[m] for m in self.sits_keys if m in days]
        if not dates_list:
            return None, None
        dates = torch.cat(dates_list, dim=1)
        sort_idx = dates.argsort(dim=1)
        sorted_dates = dates.gather(1, sort_idx)
        return sorted_dates, sort_idx
    
    @staticmethod
    def _sort_along_t(tensor, sort_idx):
        """Reorder [B, T, ...] tensor along dim=1 using sort_idx [B, T]."""
        idx = sort_idx
        for _ in range(tensor.dim() - 2):
            idx = idx.unsqueeze(-1)
        return tensor.gather(1, idx.expand_as(tensor))
 
    @staticmethod
    def _resample_temporal(data, src_dates, tgt_dates):
        """Linearly interpolate data from src_dates to tgt_dates along T.
 
        Args:
            data:      [B, T_src, C, H, W]
            src_dates: [B, T_src] — must be sorted per sample
            tgt_dates: [B, T_tgt] — target timeline
 
        Returns:
            resampled: [B, T_tgt, C, H, W]
 
        Uses linear interpolation with nearest-neighbor extrapolation at
        boundaries. Fully differentiable w.r.t. data.
        """
        assert (src_dates.diff(dim=1) >= 0).all()
        B, T_src, C, H, W = data.shape
        T_tgt = tgt_dates.shape[1]

        if T_src == 1:
            return data.expand(B, T_tgt, C, H, W).contiguous()
 
        # For each target date, find the right-bracket index in src_dates
        # searchsorted gives index where tgt would be inserted to keep sorted
        right = torch.searchsorted(src_dates, tgt_dates)      # [B, T_tgt]
        right = right.clamp(1, T_src - 1)                      # ensure valid bracket
        left = right - 1
 
        # Gather bracketing dates: [B, T_tgt]
        left_dates = src_dates.gather(1, left)
        right_dates = src_dates.gather(1, right)
 
        # Interpolation weight: how far between left and right
        denom = (right_dates - left_dates).clamp(min=1e-6)
        w = ((tgt_dates - left_dates) / denom).clamp(0, 1)    # [B, T_tgt]
 
        # Expand indices for gathering from [B, T_src, C, H, W]
        left_exp = left.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, T_tgt, C, H, W)
        right_exp = right.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, T_tgt, C, H, W)
 
        left_data = data.gather(1, left_exp)
        right_data = data.gather(1, right_exp)
 
        # Weighted blend
        w = w[:, :, None, None, None]                          # [B, T_tgt, 1, 1, 1]
        return (1 - w) * left_data + w * right_data

    # ────────────────────────────────────────────────────────────────────
    # EARLY FUSION — concat channels → single enc → single LTAE → decode
    # ────────────────────────────────────────────────────────────────────

    def _forward_early(self, image: dict, days: dict, pe: dict) -> dict:
        ref = self.early_ref_mod
        ref_dates = days[ref].contiguous()   # alignment target -- always real time

        aligned = []
        for m in self.sits_keys:
            if m == ref:
                aligned.append(image[m])
            else:
                aligned.append(self._resample_temporal(
                    image[m], days[m].contiguous(), ref_dates))

        data = torch.cat(aligned, dim=2) # (B, T_ref, C_total, H, W)

        feats = [self.in_conv.smart_forward(data)]
        for down in self.down_blocks:
            feats.append(down.smart_forward(feats[-1]))

        # PE positions: the reference modality's own fields, unresampled -- the
        # fused stream already sits exactly on ref's timeline (that is what
        # `ref_dates` is), so there is nothing to interpolate here.
        out, attn = self.temporal_encoder(
            feats[-1], batch_positions=self._pe_positions(pe, ref), pad_mask=None)
        out = self.bottleneck(out)

        dem_feats = self._encode_dem(image)
        for i, up in enumerate(self.up_blocks):
            lvl = self.n_stages - 2 - i
            skip = self.temporal_aggregator(feats[lvl], pad_mask=None, attn_mask=attn)
            out = up(out, self._cat_skip([skip], dem_feats, lvl))

        return {"segmentation": self.out_conv(out)}

 
    # ────────────────────────────────────────────────────────────────────
    # MID FUSION — per-mod spatial enc → stack along T → shared LTAE → decode
    # ────────────────────────────────────────────────────────────────────
 
    def _forward_mid(self, image: dict, days: dict, pe: dict) -> dict:
        all_feats = self._encode_sits_per_mod(image)
        # Always sorted by real elapsed time -- see TEMPORAL_PE_MODES.
        _, sort_idx = self._build_sorted_timeline(days)

        stacked = torch.cat([all_feats[m][-1] for m in self.sits_keys], dim=1)
        if sort_idx is not None:
            stacked = self._sort_along_t(stacked, sort_idx)

        out, attn = self.temporal_encoder(
            stacked, batch_positions=self._sorted_pe(pe, sort_idx), pad_mask=None)
        out = self.bottleneck(out)

        dem_feats = self._encode_dem(image)
        for i, up in enumerate(self.up_blocks):
            lvl = self.n_stages - 2 - i
            stacked_skip = torch.cat([all_feats[m][lvl] for m in self.sits_keys], dim=1)
            if sort_idx is not None:
                stacked_skip = self._sort_along_t(stacked_skip, sort_idx)
            skip = self.temporal_aggregator(stacked_skip, pad_mask=None, attn_mask=attn)
            out = up(out, self._cat_skip([skip], dem_feats, lvl))

        return {"segmentation": self.out_conv(out)}


    # ────────────────────────────────────────────────────────────────────
    # LATE FUSION — per-mod spatio-temporal enc → concat embeddings → decode
    # ────────────────────────────────────────────────────────────────────

    def _forward_late(self, image: dict, days: dict, pe: dict) -> dict:
        all_feats = self._encode_sits_per_mod(image)
        pooled, attns = self._temporal_pool_per_mod(all_feats, pe)

        fused = self.fusion_conv(torch.cat([pooled[m] for m in self.sits_keys], dim=1))

        dem_feats = self._encode_dem(image)
        out = fused
        for i, up in enumerate(self.up_blocks):
            lvl = self.n_stages - 2 - i
            skips = [self.temporal_aggregator(all_feats[m][lvl], pad_mask=None, attn_mask=attns[m])
                    for m in self.sits_keys]
            out = up(out, self._cat_skip(skips, dem_feats, lvl))

        return {"segmentation": self.out_conv(out)}

    # ────────────────────────────────────────────────────────────────────
    # DECISION FUSION — per-mod full pipeline → fuse logits
    # ────────────────────────────────────────────────────────────────────

    def _forward_decision(self, image: dict, days: dict, pe: dict) -> dict:
        all_feats = self._encode_sits_per_mod(image)
        pooled, attns = self._temporal_pool_per_mod(all_feats, pe)

        dem_feats = self._encode_dem(image)
        logits = {}
        for m in self.sits_keys:
            out = self.bottlenecks[m](pooled[m])
            for i, up in enumerate(self.up_blocks_dict[m]):
                lvl = self.n_stages - 2 - i
                skip = self.temporal_aggregator(all_feats[m][lvl], pad_mask=None, attn_mask=attns[m])
                out = up(out, self._cat_skip([skip], dem_feats, lvl))
            logits[m] = self.out_convs[m](out)

        w = F.softmax(self.decision_weights, dim=0)
        fused = sum(w[i] * logits[m] for i, m in enumerate(self.sits_keys))
        return {"segmentation": fused}

    # ────────────────────────────────────────────────────────────────────
    # NO FUSION 
    # ────────────────────────────────────────────────────────────────────

    def _forward_single(self, image: dict, days: dict, pe: dict) -> dict:
        """Single-modality path: per-mod encoder → LTAE → decode. No fusion."""
        m = self.sits_keys[0]
        feats = [self.in_convs[m].smart_forward(image[m])]
        for down in self.down_blocks[m]:
            feats.append(down.smart_forward(feats[-1]))

        out, attn = self.temporal_encoder(
            feats[-1], batch_positions=self._pe_positions(pe, m), pad_mask=None)
        out = self.bottleneck(out)

        dem_feats = self._encode_dem(image)
        for i, up in enumerate(self.up_blocks):
            lvl = self.n_stages - 2 - i
            skip = self.temporal_aggregator(feats[lvl], pad_mask=None, attn_mask=attn)
            out = up(out, self._cat_skip([skip], dem_feats, lvl))

        return {"segmentation": self.out_conv(out)}

    # ────────────────────────────────────────────────────────────────────

    def _field(self, metadata, key, device) -> dict:
        """Per-modality (B, T) tensor for a per-frame metadata field."""
        if isinstance(metadata, list):
            missing = [s["id"] for s in metadata if key not in s or not s[key]]
            if missing:
                raise KeyError(
                    f"metadata['{key}'] missing for {len(missing)} sample(s), e.g. {missing[0]!r} "
                    "-- regenerate the batch with the current dataset code.")
            return {m: torch.stack([torch.as_tensor(s[key][m]) for s in metadata], dim=0)
                       .float().to(device)
                    for m in self.sits_keys}
        return {m: metadata[key][m].float().to(device) for m in self.sits_keys}

    def _pe_positions(self, pe: dict, m: str):
        """batch_positions for one modality: a single (B, T) tensor if
        temporal_pe is one field, else a tuple of tensors -- LTAE2d sums their
        encodings. None if the modality has no entry (mirrors dict.get)."""
        vals = [d[m] for d in pe.values() if m in d]
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else tuple(vals)

    def _sorted_pe(self, pe: dict, sort_idx: torch.Tensor | None):
        """PE positions for mid fusion's merged timeline: each field
        concatenated across modalities in the same order `_build_sorted_timeline`
        used, then reordered with that SAME sort_idx -- so e.g. `doy` still
        describes the right frame after the merge, without ever being what
        decided the order (`_build_sorted_timeline` always sorts by `rel_day`)."""
        if sort_idx is None:
            return None
        out = [torch.cat([d[m] for m in self.sits_keys if m in d], dim=1).gather(1, sort_idx)
               for d in pe.values()]
        return out[0] if len(out) == 1 else tuple(out)

    def forward(self, image: dict, metadata=None) -> dict:
        device = next(iter(image.values())).device

        # Always the alignment axis -- see TEMPORAL_PE_MODES for why this is
        # fetched unconditionally, independent of temporal_pe.
        days = self._field(metadata, "rel_day", device)
        # What actually feeds the positional encoder: one field, or several
        # summed (see TEMPORAL_PE_MODES).
        pe = {f: (days if f == "rel_day" else self._field(metadata, f, device))
              for f in TEMPORAL_PE_MODES[self.temporal_pe]}

        fusion_ops = {
            "single":   self._forward_single,
            "early":    self._forward_early,
            "mid":      self._forward_mid,
            "late":     self._forward_late,
            "decision": self._forward_decision
        }
        return fusion_ops[self.fusion_stage](image, days, pe)

# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":

    B, H, W, T = 8, 256, 256, 16
    
    modality_infos = {
        "S1RTC": {"channels": 2, "temporal": True},
        "S2L2A": {"channels": 10, "temporal": True},
        "DEM": {"channels": 3, "temporal": False}
    }

    def make_batch(mod_infos):
        batch = {
            "image": {},
            "mask": {},
            "metadata": {
                "id": [f"sample_{i}" for i in range(B)],
                "timestamps": {},
                "rel_day": {},
                "date_mask": {},
            },
        }
        for m, info in mod_infos.items():
            if info.get("temporal", True):
                batch["image"][m] = torch.randn(B, T, info["channels"], H, W)
                batch["metadata"]["timestamps"][m] = torch.arange(T).float().unsqueeze(0).expand(B, -1)
                batch["metadata"]["rel_day"][m] = torch.arange(T).float().unsqueeze(0).expand(B, -1)
                batch["metadata"]["date_mask"][m] = torch.ones(B, T, dtype=torch.bool)
            else:
                batch["image"][m] = torch.randn(B, info["channels"], H, W)

        # single shared mask (adjust if you have per-modality masks)
        batch["mask"] = torch.randint(0, 2, (B, H, W), dtype=torch.long)
        return batch

    # Test all 5 strategies with 2 SITS + DEM
    for stage in (
        # "single",  # needs a single-SITS modality_infos, see assert above
        "early",
        "mid",
        "late",
        "decision",
    ):
        print(f"\n{'='*60}")
        print(f"Testing fusion_stage='{stage}'")
        print(f"{'='*60}")
        model = UTAE_MM(modality_infos, num_classes=2, fusion_stage=stage)
        batch = make_batch(modality_infos)
        out = model(batch["image"], batch["metadata"])
        print(f"  seg: {out['segmentation'].shape}")
        out['segmentation'].mean().backward()
        dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        print(f"  Dead: {dead[:5]}" if dead else "  All params receive gradients")

        print("Testing fusion WITHOUT DEM")
        modality_infos_without_dem = {k: v for k, v in modality_infos.items() if k != "DEM"}
        model = UTAE_MM(modality_infos_without_dem, num_classes=2, fusion_stage=stage)
        out = model(batch["image"], batch["metadata"])
        print(f"  seg: {out['segmentation'].shape}")
        out['segmentation'].mean().backward()
        dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        print(f"  Dead: {dead[:5]}" if dead else "  All params receive gradients")

        import gc
        del model, batch, out; gc.collect()