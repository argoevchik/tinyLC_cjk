"""Export the trained model to ONNX and check it against PyTorch with onnxruntime.

  ./run.sh -m tinyfc_cjk.export_onnx            # models/tinyfc_cjk.pt -> models/tinyfc_cjk.onnx + .json

Graph
  input   image          float32 [N, 1, 64, 256]  ink intensity: 1 = ink, 0 = paper
                         (grayscale dark-on-light u8 -> 1 - u8/255), see predict_onnx.py
  outputs coarse         [N, 4]  softmax over serif, sans, brush, rounded
          script         [N, 3]  softmax over ja, zh, ko
          fine_<lang>_<coarse>  [N, k]  softmax, one per fine head
          kana_ratio     [N]     predicted kana/kanji size ratio of the design (meaningful for ja)
Average the outputs over all windows of one line (predict_onnx.py does this).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import TRAIT_MEAN, TRAIT_STD
from .predict import MODEL, image_to_windows, load_model
from .render import H, W
from .taxonomy import COARSE, FINE_INFO, LANGS, ROOT, TRAIT_CUTS


class Exportable(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        c, s, f, t = self.m(image)
        return (c.softmax(1), s.softmax(1), *[x.softmax(1) for x in f], t * TRAIT_STD + TRAIT_MEAN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--out", default=str(ROOT / "models" / "tinyfc_cjk.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    model = load_model(a.model)
    heads = model.heads
    fine_names = [f"fine_{l}_{c}" for l, c in heads]
    out_names = ["coarse", "script", *fine_names, "kana_ratio"]
    wrapped = Exportable(model).eval()
    dummy = torch.rand(2, 1, H, W)
    torch.onnx.export(
        wrapped, (dummy,), a.out, input_names=["image"], output_names=out_names,
        dynamic_axes={"image": {0: "batch"}, **{n: {0: "batch"} for n in out_names}},
        opset_version=a.opset, dynamo=False,
    )

    meta = dict(
        input=dict(name="image", shape=[None, 1, H, W], dtype="float32",
                   encoding="grayscale, dark text on light paper, ink = 1 - pixel/255; "
                            "text line scaled to 75% of the height and centered, long lines cut into "
                            "overlapping 256-px windows (see predict_onnx.py)"),
        coarse=COARSE, script=LANGS,
        fine={n: dict(lang=l, coarse=c, classes=heads[(l, c)]) for n, (l, c) in zip(fine_names, heads)},
        fine_info={k: v for k, v in FINE_INFO.items() if any(k in h for h in heads.values())},
        kana_ratio_cuts={f"{l}/{c}": dict(cuts=v, classes=heads.get((l, c))) for (l, c), v in TRAIT_CUTS.items()},
        usage="Average each output over the windows of a line. coarse = argmax(coarse). "
              "Use the fine output whose (lang, coarse) matches argmax(script) (or the known language) "
              "and the coarse class; if none matches there is no fine class.",
    )
    import onnx

    om = onnx.load(a.out)
    om.metadata_props.add(key="tinyfc_cjk", value=json.dumps(meta, ensure_ascii=False))
    onnx.save(om, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    # --- check against PyTorch on real windows (figure lines) and noise
    import onnxruntime as ort
    from PIL import Image

    lines = sorted((ROOT / "research" / "figure_lines").glob("*.png"))[:12]
    x = np.concatenate([image_to_windows(Image.open(p)) for p in lines]) if lines else np.zeros((0, H, W), np.uint8)
    x = (1.0 - x.astype(np.float32) / 255.0)[:, None]
    x = np.concatenate([x, np.random.default_rng(0).random((4, 1, H, W), dtype=np.float32)])
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x})
    with torch.no_grad():
        ref = [t.numpy() for t in wrapped(torch.from_numpy(x))]
    diffs = {n: float(np.abs(g - r).max()) for n, g, r in zip(out_names, got, ref)}
    agree = float((got[0].argmax(1) == ref[0].argmax(1)).mean())
    size = Path(a.out).stat().st_size / 1e6
    print(f"saved {a.out} ({size:.1f} MB), opset {a.opset}, outputs {out_names}")
    print(f"onnxruntime vs torch on {len(x)} windows: max |diff| {max(diffs.values()):.2e}, coarse argmax agreement {agree:.3f}")
    assert max(diffs.values()) < 1e-3, diffs


if __name__ == "__main__":
    main()
