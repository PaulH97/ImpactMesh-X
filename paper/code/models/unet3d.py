"""
Original taken from: https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/unet3d.py
Modified by Paul Höhn, 05.05.2026
"""

import torch
import torch.nn as nn


def _norm(num_channels, num_groups=8):
    # GroupNorm with 8 groups — standard choice; falls back to LayerNorm-equivalent
    # if num_channels < num_groups.
    return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)

def conv_block(in_dim, middle_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        _norm(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        _norm(out_dim),
        nn.LeakyReLU(inplace=True),
    )


def center_in(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        _norm(out_dim),
        nn.LeakyReLU(inplace=True),
    )


def center_out(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        _norm(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
    )


def up_conv_block(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        _norm(out_dim),
        nn.LeakyReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(
        self,
        num_bands: int,
        num_classes: int,
        feats: int = 8,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_classes = num_classes

        self.en3 = conv_block(num_bands, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)
        self.final = nn.Conv3d(feats * 2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, metadata=None):
        if isinstance(inputs, dict):
            assert len(inputs) == 1, f"UNet3D expects one modality, got {list(inputs.keys())}"
            x = next(iter(inputs.values()))
        else:
            x = inputs

        # (B, T, C, H, W) -> (B, C, T, H, W)
        out = x.permute(0, 2, 1, 3, 4)

        en3 = self.en3(out)
        pool_3 = self.pool_3(en3)
        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)
        center_in = self.center_in(pool_4)
        center_out = self.center_out(center_in)
        concat4 = torch.cat([center_out, en4[:, :, : center_out.shape[2], :, :]], dim=1)
        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)
        concat3 = torch.cat([trans3, en3[:, :, : trans3.shape[2], :, :]], dim=1)
        dc3 = self.dc3(concat3)
        final = self.final(dc3)               # (B, num_classes, T', H, W)

        # collapse temporal dim -> (B, num_classes, H, W)
        logits = final.mean(dim=2)

        return {"segmentation": logits}
    
if __name__ == "__main__":

    # B, H, W, T = 8, 256, 256, 16
    B, H, W, T = 16, 128, 128, 15
    
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
                "days": {},
                "date_mask": {},
            },
        }
        for m, info in mod_infos.items():
            if info.get("temporal", True):
                batch["image"][m] = torch.randn(B, T, info["channels"], H, W)
                batch["metadata"]["timestamps"][m] = torch.arange(T).float().unsqueeze(0).expand(B, -1)
                batch["metadata"]["days"][m] = torch.arange(T).long().unsqueeze(0).expand(B, -1)
                batch["metadata"]["date_mask"][m] = torch.ones(B, T, dtype=torch.bool)
            else:
                batch["image"][m] = torch.randn(B, info["channels"], H, W)

        # single shared mask (adjust if you have per-modality masks)
        batch["mask"] = torch.randint(0, 2, (B, H, W), dtype=torch.long)
        return batch

    batch = make_batch(modality_infos)
    model = UNet3D(num_bands=modality_infos[list(modality_infos.keys())[0]]["channels"], feats=16, num_classes=2)
    output = model(batch["image"])
    print(output["segmentation"].shape)  # should be (B, num_classes, H, W)

    from fvcore.nn import FlopCountAnalysis, parameter_count
    flops = FlopCountAnalysis(model, batch["image"])
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    print(f"Total GFLOPs: {flops.total() / 1e9:.2f}")
    total_params = parameter_count(model)[""]
    print(f"Total params: {total_params / 1e6:.2f}M ")