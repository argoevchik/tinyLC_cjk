"""Classify CJK type in an image of one horizontal text line, or a font file.

  ./run.sh -m tinyfc_cjk.predict line.png [more.png ...]        # cropped text line(s)
  ./run.sh -m tinyfc_cjk.predict --lang ja line.png              # force the fine head's language
  ./run.sh -m tinyfc_cjk.predict --font Some.otf --lang ko       # render samples with a font file
  ./run.sh -m tinyfc_cjk.predict --figures                       # score the lessons' figure lines

Output: coarse class (serif / sans / brush / rounded) for any language; then, if the
line's script (predicted, or --lang) has a fine head for that coarse class, the
fine class (e.g. ja/sans -> gothic_old / gothic_humanist / gothic_geometric).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .model import TRAIT_MEAN, TRAIT_STD, TinyFC, to_tensor
from .render import H, W, Variant, fit_to_canvas, pools, render_line, sample_text, HIRAGANA, KATAKANA
from .taxonomy import COARSE, LANGS, ROOT, TRAIT_CUTS, Font, trait_class

MODEL = ROOT / "models" / "tinyfc_cjk.pt"


def load_model(path=MODEL, device="cpu"):
    ck = torch.load(path, map_location=device)
    heads = {tuple(k.split("/")): v for k, v in ck["fine_heads"].items()}
    m = TinyFC(heads)
    m.load_state_dict(ck["state_dict"])
    m.heads = heads
    return m.eval().to(device)


def image_to_windows(img: Image.Image) -> np.ndarray:
    """Photo/screenshot of one horizontal text line -> model windows."""
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
        return fit_to_canvas(line, None, text_h)[None]
    scaled = line.resize((nw, text_h), Image.Resampling.LANCZOS)
    xs0 = np.linspace(0, nw - W, num=int(np.ceil(nw / (W * 0.75))) + 1).astype(int)
    return np.stack([fit_to_canvas(scaled.crop((x0, 0, x0 + W, text_h)), None, text_h) for x0 in xs0])


@torch.no_grad()
def classify(model, windows: np.ndarray, lang=None):
    c, s, f, t = model(to_tensor(windows))
    pc = F.softmax(c, 1).mean(0).numpy()
    ps = F.softmax(s, 1).mean(0).numpy()
    lang = lang or LANGS[int(ps.argmax())]
    coarse = COARSE[int(pc.argmax())]
    out = dict(coarse=coarse, p_coarse={k: round(float(v), 3) for k, v in zip(COARSE, pc)},
               script=lang, p_script={k: round(float(v), 3) for k, v in zip(LANGS, ps)})
    key = (lang, coarse)
    if lang == "ja":  # design's kana/kanji size ratio, and the class it implies
        out["kana_ratio"] = round(float((t * TRAIT_STD + TRAIT_MEAN).median()), 3)
        if key in TRAIT_CUTS:
            out["fine_by_ratio"] = trait_class(key, out["kana_ratio"])
    if key in model.heads:
        k = list(model.heads).index(key)
        pf = F.softmax(f[k], 1).mean(0).numpy()
        out["fine"] = model.heads[key][int(pf.argmax())]
        out["p_fine"] = {n: round(float(v), 3) for n, v in zip(model.heads[key], pf)}
    return out


def report(name, r):
    fine = f"  ->  {r['fine']} ({max(r['p_fine'].values()):.2f})" if "fine" in r else ""
    print(f"{name}\n  {r['coarse']} ({r['p_coarse'][r['coarse']]:.2f})  script={r['script']}{fine}")
    print("    coarse " + "  ".join(f"{k} {v:.2f}" for k, v in r["p_coarse"].items()))
    if "p_fine" in r:
        print("    fine   " + "  ".join(f"{k} {v:.2f}" for k, v in r["p_fine"].items()))
    if "kana_ratio" in r:
        print(f"    kana/kanji size ratio {r['kana_ratio']}" + (f"  -> {r['fine_by_ratio']}" if "fine_by_ratio" in r else ""))


def font_windows(path, lang, n=64, wght=None):
    from fontTools.ttLib import TTFont

    cm = set(TTFont(path, fontNumber=0, lazy=True).getBestCmap())
    has = lambda cs: "".join(c for c in cs if ord(c) in cm)
    P = pools()
    cs = dict(kana=has(HIRAGANA + KATAKANA), jis=has(P["jis"]), gb=has(P["gb"]), big5=has(P["big5"]),
              hangul=has(P["hangul"]))
    cs = {k: v for k, v in cs.items() if v}
    v = Variant(Font("cli", "sans", "cli", (lang,)), Path(path), wght)
    rng = random.Random(0)
    return np.stack([fit_to_canvas(render_line(v, sample_text(cs, lang, rng, 6)), None, int(H * 0.75)) for _ in range(n)])


def score_figures(model):
    labels = json.loads((ROOT / "research" / "figure_labels.json").read_text())
    by_font = {}
    for fn, lab in labels.items():
        wins = image_to_windows(Image.open(ROOT / "research" / "figure_lines" / fn))
        by_font.setdefault((lab["font"], fn.split("_")[0]), (lab, []))[1].append(wins)
    rows = []
    for (font, lang_name), (lab, wins) in by_font.items():
        lang = {"chinese": "zh", "japanese": "ja", "korean": "ko"}[lang_name]
        r = classify(model, np.concatenate(wins), None)  # language not given: routed by the script head
        ok_c = None if lab["coarse"] is None else r["coarse"] == lab["coarse"]
        ok_f = None if not lab["fine"] or "fine" not in r else r["fine"] == lab["fine"]
        rows.append(dict(font=font, lang=lang, seen=lab["in_training"], true=lab["coarse"], pred=r["coarse"],
                         p=r["p_coarse"][r["coarse"]], ok=ok_c, fine_true=lab["fine"], fine_pred=r.get("fine"),
                         p_fine=r.get("p_fine"), fine_ok=ok_f, kana_ratio=r.get("kana_ratio"),
                         fine_by_ratio=r.get("fine_by_ratio"),
                         ratio_ok=None if not lab["fine"] or "fine_by_ratio" not in r else r["fine_by_ratio"] == lab["fine"], script_pred=max(r["p_script"], key=r["p_script"].get),
                         note=lab["note"]))
    mark = lambda b: "  " if b is None else ("OK" if b else "XX")
    for x in rows:
        fine = f" | fine {mark(x['fine_ok'])} {x['fine_true'] or '-'} -> {x['fine_pred'] or '-'}" if x["fine_true"] or x["fine_pred"] else ""
        if x["fine_by_ratio"]:
            fine += f" | ratio {x['kana_ratio']} {mark(x['ratio_ok'])} {x['fine_by_ratio']}"
        print(f"{mark(x['ok'])} {x['lang']} {'seen  ' if x['seen'] else 'UNSEEN'} {x['font']:24s} {str(x['true']):8s} -> {x['pred']:8s} {x['p']:.2f}{fine}  script:{x['script_pred']}")
    for seen in (False, True):
        s = [x for x in rows if x["seen"] == seen and x["ok"] is not None]
        f = [x for x in rows if x["seen"] == seen and x["fine_ok"] is not None]
        q = [x for x in rows if x["seen"] == seen and x["ratio_ok"] is not None]
        print(f"{'seen' if seen else 'unseen'} fonts: coarse {sum(x['ok'] for x in s)}/{len(s)}, fine head {sum(x['fine_ok'] for x in f)}/{len(f)}, fine by ratio {sum(x['ratio_ok'] for x in q)}/{len(q)}")
    out = ROOT / "results" / "figures.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--font", action="append", default=[])
    ap.add_argument("--wght", type=float, default=None)
    ap.add_argument("--lang", choices=LANGS, default=None, help="script of the text (default: predicted)")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    model = load_model(a.model)
    if a.figures:
        score_figures(model)
    for p in a.images:
        report(p, classify(model, image_to_windows(Image.open(p)), a.lang))
    for fp in a.font:
        report(fp, classify(model, font_windows(fp, a.lang or "ja", wght=a.wght), a.lang or "ja"))


if __name__ == "__main__":
    main()
