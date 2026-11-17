"""Prediction galleries for the README (docs/*.png). The language is never given:
the model routes to a fine head with its own script prediction.

  ./run.sh -m tinyfc_cjk.gallery

  docs/figures_{zh,ja,ko}.png   lines cut from the lessons' figures, final model
  docs/heldout_{ja,zh,ko}.png   one sample line per font, with the prediction the
                                cross-validation model made while that font's design
                                group was held out (results/cv_auto.json, 120 crops)
Green bar = coarse and (if any) fine class right, red = wrong, grey = no label.
"""

import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .predict import classify, image_to_windows, load_model
from .render import charsets, discover_variants, render_line
from .taxonomy import FONTS, ROOT

DOCS = ROOT / "docs"
UI = ROOT / "fonts" / "ibmplexsansjp"
GREEN, RED, GREY = (46, 160, 67), (207, 34, 46), (150, 150, 150)
ROW_H, COL_W, IMG_W, IMG_H = 64, 760, 330, 40
SAMPLE = {"ja": "オールド書体はどんな？ゴシック", "sc": "楷书是从隶书演变第一种印刷", "tc": "體就是宋體或明體印刷和雕版",
          "ko": "한글 세리프 비교 가나다라"}


def ui(size, bold=False):
    return ImageFont.truetype(str(UI / ("IBMPlexSansJP-SemiBold.ttf" if bold else "IBMPlexSansJP-Regular.ttf")), size)


def fit(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    s = IMG_H / img.height
    img = img.resize((max(1, round(img.width * s)), IMG_H), Image.Resampling.LANCZOS)
    return img.crop((0, 0, min(img.width, IMG_W), IMG_H))


def sheet(rows, title, path, cols=2):
    """rows: (image, name, true_text, pred_text, colour)."""
    n = (len(rows) + cols - 1) // cols
    W, H = COL_W * cols + 16, 52 + ROW_H * n + 8
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    d.text((12, 12), title, font=ui(22, True), fill=(30, 30, 30))
    for i, (img, name, true, pred, colour) in enumerate(rows):
        x, y = 8 + COL_W * (i // n), 52 + ROW_H * (i % n)
        if (i % n) % 2 == 0:
            d.rectangle((x, y, x + COL_W - 8, y + ROW_H - 1), fill=(246, 247, 249))
        d.rectangle((x, y + 4, x + 5, y + ROW_H - 5), fill=colour)
        out.paste(fit(img), (x + 14, y + (ROW_H - IMG_H) // 2))
        tx = x + 14 + IMG_W + 14
        d.text((tx, y + 6), name, font=ui(15, True), fill=(30, 30, 30))
        d.text((tx, y + 25), "true  " + true, font=ui(13), fill=(90, 90, 90))
        d.text((tx, y + 42), "pred  " + pred, font=ui(13), fill=colour if colour != GREY else (60, 60, 60))
    DOCS.mkdir(exist_ok=True)
    out.save(path, optimize=True)
    print("wrote", path.relative_to(ROOT), len(rows), "rows")


def figures(model):
    labels = json.loads((ROOT / "research" / "figure_labels.json").read_text())
    by = defaultdict(list)
    for fn, lab in labels.items():
        by[(fn.split("_")[0], lab["font"])].append((fn, lab))
    res = {}
    for (lang_name, font), items in by.items():
        lab = items[0][1]
        imgs = [Image.open(ROOT / "research" / "figure_lines" / fn) for fn, _ in items]
        r = classify(model, np.concatenate([image_to_windows(im) for im in imgs]), None)  # language not given
        pred = r["coarse"] + (f" / {r['fine']}" if "fine" in r else "") + f"   {r['p_coarse'][r['coarse']]:.2f}" \
            + f"   script {r['script']}"
        if lab["coarse"] is None:
            colour, true = GREY, "decorative (outside the 4 classes)"
        else:
            ok = r["coarse"] == lab["coarse"] and (not lab["fine"] or r.get("fine") == lab["fine"])
            colour = GREEN if ok else RED
            true = lab["coarse"] + (f" / {lab['fine']}" if lab["fine"] else "")
        name = font + ("" if lab["in_training"] else "  · unseen")
        res.setdefault(lang_name, []).append((imgs[0], name, true, pred, colour))
    titles = {"chinese": "Chinese lesson figures", "japanese": "Japanese lesson figures", "korean": "Korean lesson figures"}
    short = {"chinese": "zh", "japanese": "ja", "korean": "ko"}
    for lang_name, rows in res.items():
        sheet(rows, titles[lang_name] + " — final model, language not given", DOCS / f"figures_{short[lang_name]}.png",
              cols=1)


def pick_face(variants):
    for v in variants:
        if v.wght == 400 or "Regular" in v.path.name:
            return v
    return variants[0]


def heldout():
    cv = json.loads((ROOT / "results" / "cv_auto.json").read_text())
    preds = {k: v for fold in cv.values() for k, v in fold["fonts"].items()}
    cs = charsets()
    faces = defaultdict(list)
    for v in discover_variants():
        faces[v.font.id].append(v)
    names = {"ja": "Japanese", "zh": "Chinese", "ko": "Korean"}
    for lang in ("ja", "zh", "ko"):
        rows = []
        for f in FONTS:
            r = preds.get(f"{f.id}/{lang}")
            if r is None:
                continue
            v = pick_face(faces[f.id])
            c = cs[v.key]
            key = lang if lang != "zh" else ("sc" if "gb" in c else "tc")
            have = set("".join(x for k, x in c.items() if not k.startswith("punct")))
            text = "".join(ch for ch in SAMPLE[key] if ch in have or ch == " ") or SAMPLE[key]
            img = render_line(v, text)
            fine_true, fine_pred = r.get("fine_true"), r.get("auto_fine")
            ok = r["pred"] == r["true"] and (fine_true is None or fine_pred == fine_true)
            true = r["true"] + (f" / {fine_true}" if fine_true else "")
            pred = r["pred"] + (f" / {fine_pred}" if fine_pred else "") + f"   {r['p']:.2f}" + f"   script {r['script_pred']}"
            if "ratio_pred" in r:
                pred += f"   kana {r['ratio_pred']:.2f}"
            rows.append((img, f.id, true, pred, GREEN if ok else RED))
        n_ok = sum(r[4] == GREEN for r in rows)
        sheet(rows, f"{names[lang]}: unseen designs (cross-validation, held out), language not given — {n_ok}/{len(rows)} fully right",
              DOCS / f"heldout_{lang}.png")


def main():
    figures(load_model())
    heldout()


if __name__ == "__main__":
    main()
