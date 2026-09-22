"""
Original taken from: https://github.com/michaeltrs/DeepSatModels/blob/main/models/TSViT/TSViTdense.py
Modified by Paul Höhn
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Deliberately the same encoder U-TAE uses, not a copy: both models must see
# acquisition time encoded identically, or the benchmark compares encodings
# instead of architectures.
from fusion.models.utae import PositionalEncoder, TEMPORAL_PE_MODES


# =============================================================================
# Transformer building blocks (vanilla, vendored to drop external deps)
# =============================================================================

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# =============================================================================
# TSViT
# =============================================================================

class TSViT(nn.Module):
    """
    Temporal-Spatial ViT for single-modality SITS semantic segmentation.

    Forward expects batch["image"] = {modality: (B, T, C, H, W)} with exactly
    one key (or a raw tensor of that shape).

    Returns {"segmentation": (B, num_classes, H, W)}.
    """

    def __init__(
        self,
        num_bands: int,
        num_classes: int,
        img_size: int,
        max_seq_len: int,
        patch_size: int = 2,
        dim: int = 128,
        temporal_depth: int = 6,
        spatial_depth: int = 2,
        heads: int = 4,
        dim_head: int = 64,
        scale_dim: int = 4,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        temporal_pe: str = "relative",
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.image_size = img_size
        self.patch_size = patch_size
        self.num_patches_1d = img_size // patch_size
        self.num_classes = num_classes
        self.num_frames = max_seq_len
        self.dim = dim
        assert temporal_pe in TEMPORAL_PE_MODES, (
            f"temporal_pe must be one of {sorted(TEMPORAL_PE_MODES)}, got {temporal_pe!r}")
        self.temporal_pe = temporal_pe

        num_patches = self.num_patches_1d ** 2
        patch_dim = num_bands * patch_size ** 2

        # Patch embedding — folds spatial patches into tokens, keeps T as a sequence dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        # The same sinusoidal encoder as in UTAE_MM -- same class, same
        # parametrisation. Only this way does "TSViT vs U-TAE" compare
        # architectures rather than two notions of time: a learned index
        # embedding here and sinusoidal days there would hand the two models
        # different amounts of information about the same series. The original
        # TSViT encodes real dates, not sequence position, so this is also the
        # faithful variant.
        self.positional_encoder = PositionalEncoder(dim)

        # One cls token per class (TSViT key idea: temporal classifier per class)
        self.temporal_token = nn.Parameter(torch.randn(1, num_classes, dim))

        self.temporal_transformer = Transformer(
            dim, temporal_depth, heads, dim_head, dim * scale_dim, dropout
        )
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.space_transformer = Transformer(
            dim, spatial_depth, heads, dim_head, dim * scale_dim, dropout
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_size ** 2),
        )

    @staticmethod
    def _field(metadata, key, mod, device):
        """(B, T) tensor for a per-frame metadata field, or None if unavailable."""
        def pick(container):
            if isinstance(container, dict):
                return container[mod] if mod is not None else next(iter(container.values()))
            return container

        if isinstance(metadata, list):
            return torch.stack([
                torch.as_tensor(pick(s[key]), dtype=torch.float32) for s in metadata]).to(device)
        return torch.as_tensor(pick(metadata[key]), dtype=torch.float32).to(device)

    def _positions(self, metadata, mod, B, T, device):
        """Positions the temporal PE runs on: one tensor per field in
        TEMPORAL_PE_MODES[self.temporal_pe]. Mirrors UTAE_MM._pe_positions."""
        out = []
        for f in TEMPORAL_PE_MODES[self.temporal_pe]:
            pos = self._field(metadata, f, mod, device)
            assert pos.shape == (B, T), f"expected positions of shape {(B, T)}, got {tuple(pos.shape)}"
            out.append(pos)
        return out

    def forward(self, inputs, metadata=None):
        mod = None
        if isinstance(inputs, dict):
            assert len(inputs) == 1, f"TSViT expects one modality, got {list(inputs.keys())}"
            mod, x = next(iter(inputs.items()))
        else:
            x = inputs

        B, T, C, H, W = x.shape
        assert T <= self.num_frames, (
            f"Got T={T} but model was built with max_seq_len={self.num_frames}. "
            f"Increase max_seq_len in the model config."
        )

        # 1) Spatial patchify -> (B*N_patches, T, dim)
        x = self.to_patch_embedding(x)               # (B*N, T, dim)
        x = x.reshape(B, -1, T, self.dim)            # (B, N, T, dim)

        # Sum one sinusoidal encoding per field -- see UTAE_MM's LTAE2d.forward
        # for why summing is free (PositionalEncoder has no parameters).
        pe = sum(self.positional_encoder(p)
                 for p in self._positions(metadata, mod, B, T, x.device)).unsqueeze(1)  # (B, 1, T, dim)
        x = x + pe.to(x.dtype)                       # broadcast over batch + N

        x = x.reshape(-1, T, self.dim)               # (B*N, T, dim)

        # 2) Prepend per-class temporal cls tokens, run temporal transformer
        cls_tokens = repeat(self.temporal_token, "() N d -> b N d", b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_tokens, x), dim=1)        # (B*N, K+T, dim)
        x = self.temporal_transformer(x)
        x = x[:, : self.num_classes]                 # (B*N, K, dim)

        # 3) Reshape so each class becomes its own batch of spatial tokens, run spatial transformer
        x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim)
        x = x.permute(0, 2, 1, 3).reshape(
            B * self.num_classes, self.num_patches_1d ** 2, self.dim
        )                                            # (B*K, N, dim)
        x = x + self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)                # (B*K, N, dim)

        # 4) Project each spatial token back to patch_size^2 pixels
        x = self.mlp_head(x.reshape(-1, self.dim))   # (B*K*N, P^2)
        x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2)

        # 5) Unfold patches back into the image grid
        x = rearrange(
            x,
            "b k (h w) (p1 p2) -> b k (h p1) (w p2)",
            h=self.num_patches_1d, w=self.num_patches_1d,
            p1=self.patch_size, p2=self.patch_size,
        )                                            # (B, K, H, W)

        return {"segmentation": x}

if __name__ == "__main__":

    # B, H, W, T = 8, 256, 256, 16
    B, H, W, T = 8, 128, 128, 15
    
    modality_infos = {
        # "S1RTC": {"channels": 2, "temporal": True},
        "S2L2A": {"channels": 10, "temporal": True},
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

    batch = make_batch(modality_infos)
    model = TSViT(
        num_bands=modality_infos[list(modality_infos.keys())[0]]["channels"],
        num_classes=2,
        img_size=H,
        max_seq_len=T,
        patch_size=4
    )
    output = model(batch["image"])
    print(output["segmentation"].shape)  # should be (B, num_classes, H, W)

    from fvcore.nn import FlopCountAnalysis, parameter_count
    flops = FlopCountAnalysis(model, batch["image"])
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    print(f"Total GFLOPs: {flops.total() / 1e9:.2f}")
    total_params = parameter_count(model)[""]
    print(f"Total params: {total_params / 1e6:.2f}M ")