"""
Original from https://github.com/TUM-LMF/MTLCC-pytorch/blob/master/src/models/convlstm/convlstm.py
Modified by Paul Höhn
"""
import torch
import torch.nn as nn
 
 
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
 
    def forward(self, x, state):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
 
 
class ConvLSTM(nn.Module):
    """
    Stacked ConvLSTM segmentation model.
 
    Forward expects a dict {modality: (B, T, C, H, W)} with exactly one key,
    or a raw tensor of the same shape. Returns {"segmentation": (B, num_classes, H, W)}.
    """
 
    def __init__(
        self,
        height: int,
        width: int,
        num_bands: int,
        num_classes: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.height = height
        self.width = width
 
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        if isinstance(kernel_size, int):
            kernel_size = [(kernel_size, kernel_size)] * num_layers
        kernel_size = [tuple(k) for k in kernel_size]
        assert len(hidden_dim) == len(kernel_size) == num_layers, "Inconsistent layer config."
 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
     
        cells = []
        for i in range(num_layers):
            in_dim = num_bands if i == 0 else hidden_dim[i - 1]
            cells.append(ConvLSTMCell(in_dim, hidden_dim[i], kernel_size[i], bias))
        self.cell_list = nn.ModuleList(cells)
 
        self.head = nn.Conv2d(hidden_dim[-1], num_classes, kernel_size=1)
 
    def forward(self, inputs, metadata=None):
        if isinstance(inputs, dict):
            assert len(inputs) == 1, f"ConvLSTM expects one modality, got {list(inputs.keys())}"
            x = next(iter(inputs.values()))
        else:
            x = inputs
 
        B, T = x.size(0), x.size(1)
        device, dtype = x.device, x.dtype
 
        cur = x
        for layer_idx, cell in enumerate(self.cell_list):
            h = torch.zeros(B, self.hidden_dim[layer_idx], self.height, self.width, device=device, dtype=dtype)
            c = torch.zeros_like(h)
            outputs = []
            for t in range(T):
                h, c = cell(cur[:, t], (h, c))
                outputs.append(h)
            cur = torch.stack(outputs, dim=1)
 
        logits = self.head(cur[:, -1])
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
    model = ConvLSTM(height=H, width=W, num_bands=modality_infos[list(modality_infos.keys())[0]]["channels"], num_classes=2)
    output = model(batch["image"])
    print(output["segmentation"].shape)  # should be (B, num_classes, H, W)

    from fvcore.nn import FlopCountAnalysis, parameter_count
    flops = FlopCountAnalysis(model, batch["image"])
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    print(f"Total GFLOPs: {flops.total() / 1e9:.2f}")
    total_params = parameter_count(model)[""]
    print(f"Total params: {total_params / 1e6:.2f}M ")