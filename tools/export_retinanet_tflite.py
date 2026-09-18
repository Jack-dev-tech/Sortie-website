"""
Convert an mmdet/IceVision RetinaNet (ResNet-50 FPN) checkpoint (.pth) to a
self-contained .tflite that Sortie's model.py can run.

    python tools/export_retinanet_tflite.py ~/Downloads/garbageclassification.pth model.tflite

Needs a Python env with:  pip install torch torchvision litert-torch ai-edge-litert pillow
(This is NOT the project venv; TFLite conversion needs TensorFlow, which is
heavy. A one-off env, e.g. `python3.11 -m venv ~/tflite-export`, is fine.)

What the exported model looks like (model.py auto-detects this layout):
    input   uint8 [1, H, W, 3]  RGB image
    output  float32 [1, N, 4]   boxes as (ymin, xmin, ymax, xmax), normalized 0..1
    output  float32 [1, N, C]   per-class sigmoid scores (no background class)
plus the class names embedded in the file, so model.py finds the labels itself.

The network is re-implemented here in plain PyTorch (mmdet's RetinaNet config
`retinanet_r50_fpn_1x`: pytorch-style ResNet-50, FPN with P6/P7 from C5,
4-conv RetinaHead, 9 anchors = 3 scales x 3 ratios, delta box coder) so that
mmcv/mmdet need not be installed. Weights load 1:1 from the checkpoint.
"""

import io
import math
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Anchor settings from mmdet's retinanet_r50_fpn config.
STRIDES = [8, 16, 32, 64, 128]
OCTAVE_BASE_SCALE = 4
SCALES_PER_OCTAVE = 3
RATIOS = [0.5, 1.0, 2.0]
NUM_ANCHORS = SCALES_PER_OCTAVE * len(RATIOS)
BBOX_CLIP = math.log(1000.0 / 16)          # mmdet's wh_ratio_clip
MEAN = [0.485, 0.456, 0.406]               # ImageNet, RGB (IceVision Normalize)
STD = [0.229, 0.224, 0.225]


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1)


class FPN(nn.Module):
    """mmdet FPN: start_level=1, add_extra_convs='on_input', num_outs=5, no relu before extras."""

    def __init__(self, in_channels=(512, 1024, 2048), out=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(nn.Conv2d(c, out, 1) for c in in_channels)
        self.fpn_convs = nn.ModuleList([conv3x3(out, out) for _ in in_channels]
                                       + [conv3x3(in_channels[-1], out, 2), conv3x3(out, out, 2)])

    def forward(self, c3, c4, c5):
        lat = [l(x) for l, x in zip(self.lateral_convs, (c3, c4, c5))]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        outs = [self.fpn_convs[i](lat[i]) for i in range(3)]
        p6 = self.fpn_convs[3](c5)
        p7 = self.fpn_convs[4](p6)
        return outs + [p6, p7]


class RetinaHead(nn.Module):
    def __init__(self, num_classes, feat=256, stacked=4):
        super().__init__()
        self.cls_convs = nn.ModuleList(conv3x3(feat, feat) for _ in range(stacked))
        self.reg_convs = nn.ModuleList(conv3x3(feat, feat) for _ in range(stacked))
        self.retina_cls = conv3x3(feat, NUM_ANCHORS * num_classes)
        self.retina_reg = conv3x3(feat, NUM_ANCHORS * 4)
        self.num_classes = num_classes

    def forward(self, feats):
        cls_out, reg_out = [], []
        for x in feats:
            c = r = x
            for conv in self.cls_convs:
                c = F.relu(conv(c))
            for conv in self.reg_convs:
                r = F.relu(conv(r))
            cls = self.retina_cls(c)                      # (1, A*C, H, W)
            reg = self.retina_reg(r)                      # (1, A*4, H, W)
            cls_out.append(cls.permute(0, 2, 3, 1).reshape(1, -1, self.num_classes))
            reg_out.append(reg.permute(0, 2, 3, 1).reshape(1, -1, 4))
        return torch.cat(cls_out, 1), torch.cat(reg_out, 1)


def make_anchors(height, width):
    """All anchors for an input of (height, width), mmdet ordering: level, y, x, ratio, scale."""
    all_anchors = []
    scales = torch.tensor([OCTAVE_BASE_SCALE * 2 ** (i / SCALES_PER_OCTAVE) for i in range(SCALES_PER_OCTAVE)])
    ratios = torch.tensor(RATIOS)
    h_ratios, w_ratios = torch.sqrt(ratios), 1 / torch.sqrt(ratios)
    for s in STRIDES:
        ws = (s * w_ratios[:, None] * scales[None, :]).reshape(-1)
        hs = (s * h_ratios[:, None] * scales[None, :]).reshape(-1)
        base = torch.stack([-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1)          # (A, 4) x1 y1 x2 y2
        fh, fw = math.ceil(height / s), math.ceil(width / s)
        sx = torch.arange(fw) * s
        sy = torch.arange(fh) * s
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        shifts = torch.stack([xx.reshape(-1), yy.reshape(-1)] * 2, dim=1).float()   # (HW, 4)
        all_anchors.append((base[None] + shifts[:, None]).reshape(-1, 4))
    return torch.cat(all_anchors, 0)


class RetinaNetTFLite(nn.Module):
    """uint8 NHWC image in -> (normalized boxes, sigmoid scores) out."""

    def __init__(self, num_classes, height, width):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.neck = FPN()
        self.bbox_head = RetinaHead(num_classes)
        self.height, self.width = height, width
        self.register_buffer("anchors", make_anchors(height, width))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, image_u8):
        x = image_u8.to(torch.float32).permute(0, 3, 1, 2)                 # NHWC -> NCHW
        x = (x - self.mean) / self.std
        x = self.stem(x)
        c2 = self.layer1(x); c3 = self.layer2(c2); c4 = self.layer3(c3); c5 = self.layer4(c4)
        cls, deltas = self.bbox_head(self.neck(c3, c4, c5))
        boxes = self.decode(deltas[0])
        return boxes[None], torch.sigmoid(cls)

    def decode(self, d):
        a = self.anchors
        pw, ph = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
        px, py = a[:, 0] + 0.5 * pw, a[:, 1] + 0.5 * ph
        dw = d[:, 2].clamp(-BBOX_CLIP, BBOX_CLIP)
        dh = d[:, 3].clamp(-BBOX_CLIP, BBOX_CLIP)
        gx, gy = px + pw * d[:, 0], py + ph * d[:, 1]
        gw, gh = pw * torch.exp(dw), ph * torch.exp(dh)
        x1 = (gx - 0.5 * gw).clamp(0, self.width) / self.width
        y1 = (gy - 0.5 * gh).clamp(0, self.height) / self.height
        x2 = (gx + 0.5 * gw).clamp(0, self.width) / self.width
        y2 = (gy + 0.5 * gh).clamp(0, self.height) / self.height
        return torch.stack([y1, x1, y2, x2], dim=1)


def remap_state_dict(sd):
    """mmdet key names -> this module's names."""
    out = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("backbone."):
            nk = k[len("backbone."):]
            if nk.startswith(("conv1.", "bn1.")):
                idx = "0" if nk.startswith("conv1.") else "1"
                nk = f"stem.{idx}." + nk.split(".", 1)[1]
        elif k.startswith("neck."):
            nk = k.replace(".conv.", ".")
        elif k.startswith("bbox_head."):
            nk = k.replace(".conv.", ".")
        out[nk] = v
    return out


def embed_labels(tflite_bytes, labels):
    """Append a zip holding labels.txt (the TFLite metadata convention) so model.py can read them."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("labels.txt", "\n".join(labels) + "\n")
    return tflite_bytes + buf.getvalue()


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    height = width = int(sys.argv[3]) if len(sys.argv) > 3 else 224

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta, sd = ck.get("meta", {}), ck["state_dict"]
    classes = [c for c in meta.get("classes", []) if c != "background"]
    num_classes = sd["bbox_head.retina_cls.weight"].shape[0] // NUM_ANCHORS
    if len(classes) != num_classes:
        print(f"[warn] checkpoint lists {len(classes)} classes but head has {num_classes}; using class_i")
        classes = [f"class_{i}" for i in range(num_classes)]
    print(f"classes ({num_classes}): {classes}   trained img_size={meta.get('img_size')}   export size={height}x{width}")

    model = RetinaNetTFLite(num_classes, height, width).eval()
    missing, unexpected = model.load_state_dict(remap_state_dict(sd), strict=False)
    missing = [m for m in missing if not m.endswith(("anchors", "mean", "std"))]
    if missing or unexpected:
        print("missing:", missing); print("unexpected:", unexpected)
        raise SystemExit("weight names did not line up; refusing to export a broken model")
    print("weights loaded 1:1")

    sample = (torch.zeros(1, height, width, 3, dtype=torch.uint8),)
    with torch.no_grad():
        ref_boxes, ref_scores = model(*sample)
    print(f"torch output: boxes {tuple(ref_boxes.shape)}  scores {tuple(ref_scores.shape)}")

    try:
        import litert_torch as converter          # current name
    except ImportError:
        import ai_edge_torch as converter         # older name of the same package
    edge = converter.convert(model, sample)
    edge.export(out_path)

    data = embed_labels(Path(out_path).read_bytes(), classes)
    Path(out_path).write_bytes(data)
    print(f"wrote {out_path}  ({len(data)/1e6:.1f} MB)  with embedded labels")

    # Sanity: run the .tflite on a random image and compare against torch.
    from ai_edge_litert.interpreter import Interpreter
    import numpy as np
    rnd = (torch.rand(1, height, width, 3) * 255).to(torch.uint8)
    with torch.no_grad():
        tb, ts = model(rnd)
    it = Interpreter(model_path=out_path); it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], rnd.numpy()); it.invoke()
    outs = {tuple(o["shape"][1:]): it.get_tensor(o["index"]) for o in it.get_output_details()}
    lb = outs[tuple(tb.shape[1:])]; ls = outs[tuple(ts.shape[1:])]
    print(f"max |torch - tflite|: boxes {np.abs(lb - tb.numpy()).max():.2e}  scores {np.abs(ls - ts.numpy()).max():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
