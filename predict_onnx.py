"""Standalone ONNX inference: needs only numpy, Pillow and onnxruntime (no torch, no fonts).

  python predict_onnx.py line.png [more.png ...] [--lang ja|zh|ko] [--model models/tinyfc_cjk.onnx]

Input: an image of ONE horizontal text line (crop it first). Output: coarse class
(serif / sans / brush / rounded), predicted script, and the fine class when one exists.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

H, W = 64, 256
HERE = Path(__file__).resolve().parent


def fit(line: Image.Image, text_h: int) -> np.ndarray:
    nw = max(1, round(line.width * text_h / max(line.height, 1)))
    line = line.resize((nw, text_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (W, H), 255)
    if nw > W:
        line, nw = line.crop((0, 0, W, text_h)), W
    canvas.paste(line, ((W - nw) // 2, (H - text_h) // 2))
    return np.asarray(canvas)


def windows(img: Image.Image) -> np.ndarray:
    """Same preprocessing as tinyfc_cjk.predict.image_to_windows -> float32 [N,1,64,256], ink = 1."""
    g = np.asarray(ImageOps.grayscale(img.convert("RGB")), dtype=np.float32)
    if np.median(g) < 128:  # light text on dark background
        g = 255 - g
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    g = np.clip((g - lo) / max(hi - lo, 1) * 255, 0, 255)
    ys, xs = np.where(g < 128)
    if len(xs):
        g = g[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    line = Image.fromarray(g.astype(np.uint8))
    text_h = int(H * 0.75)
    nw = round(line.width * text_h / line.height)
    if nw <= W:
        wins = fit(line, text_h)[None]
    else:
        scaled = line.resize((nw, text_h), Image.Resampling.LANCZOS)
        x0s = np.linspace(0, nw - W, num=int(np.ceil(nw / (W * 0.75))) + 1).astype(int)
        wins = np.stack([fit(scaled.crop((x0, 0, x0 + W, text_h)), text_h) for x0 in x0s])
    return (1.0 - wins.astype(np.float32) / 255.0)[:, None]


def classify(sess, meta, img, lang=None):
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, sess.run(None, {"image": windows(img)})))
    coarse_p = out["coarse"].mean(0)
    script_p = out["script"].mean(0)
    coarse = meta["coarse"][int(coarse_p.argmax())]
    lang = lang or meta["script"][int(script_p.argmax())]
    r = dict(coarse=coarse, p=round(float(coarse_p.max()), 3), script=lang)
    for name, h in meta["fine"].items():
        if (h["lang"], h["coarse"]) == (lang, coarse):
            fp = out[name].mean(0)
            r["fine"] = h["classes"][int(fp.argmax())]
            r["p_fine"] = round(float(fp.max()), 3)
    if lang == "ja":
        r["kana_ratio"] = round(float(np.median(out["kana_ratio"])), 3)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--lang", choices=["ja", "zh", "ko"])
    ap.add_argument("--model", default=str(HERE / "models" / "tinyfc_cjk.onnx"))
    a = ap.parse_args()
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    meta = json.loads(sess.get_modelmeta().custom_metadata_map["tinyfc_cjk"])
    for p in a.images:
        print(p, classify(sess, meta, Image.open(p), a.lang))


if __name__ == "__main__":
    main()
