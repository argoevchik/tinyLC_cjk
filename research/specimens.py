"""Coverage table + specimen sheets for every downloaded CJK family (for labelling by eye)."""

import json
import re
import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "fonts"
OUT = ROOT / "research" / "specimens"
OUT.mkdir(parents=True, exist_ok=True)

TEXT = {
    "ja": "オールド書体はどんな？ゴシック書体でモダン あい永語",
    "ko": "한글 세리프 비교 가나다라 반가워요 사랑해요",
    "sc": "楷书是从隶书演变 第一种印刷的新字 黑体在汉字",
    "tc": "體就是宋體或明體 印刷和雕版工藝 黑體在漢字",
}


def gb_level1():
    out = []
    for b1 in range(0xB0, 0xD8):
        for b2 in range(0xA1, 0xFF):
            try:
                out.append(ord(bytes([b1, b2]).decode("gb2312")))
            except UnicodeDecodeError:
                pass
    return set(out)


def big5_common():
    out = []
    for b1 in range(0xA4, 0xC7):
        for b2 in list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF)):
            try:
                out.append(ord(bytes([b1, b2]).decode("big5")))
            except UnicodeDecodeError:
                pass
    return set(out)


def ksx1001_hangul():
    out = []
    for b1 in range(0xB0, 0xC9):
        for b2 in range(0xA1, 0xFF):
            try:
                out.append(ord(bytes([b1, b2]).decode("euc_kr")))
            except UnicodeDecodeError:
                pass
    return set(out)


SETS = dict(
    kana=set(range(0x3041, 0x3097)) | set(range(0x30A1, 0x30FB)),
    hangul=ksx1001_hangul(),
    gb1=gb_level1(),
    big5=big5_common(),
    jis1=None,  # filled below
)
# JIS X 0208 level 1 kanji
_j = []
for b1 in range(0xB0, 0xD0):
    for b2 in range(0xA1, 0xFF):
        try:
            _j.append(ord(bytes([b1, b2]).decode("euc_jp")))
        except UnicodeDecodeError:
            pass
SETS["jis1"] = set(_j)


def pick_file(d: Path):
    files = sorted(d.glob("*.ttf"))
    if not files:
        return None
    for key in ("[", "-Regular.ttf", "Regular", "-Medium"):
        for f in files:
            if key in f.name and "Italic" not in f.name:
                return f
    return files[0]


def main():
    rows = {}
    for d in sorted(FONTS.iterdir()):
        f = pick_file(d)
        if f is None:
            continue
        cmap = set(TTFont(f, lazy=True).getBestCmap())
        cov = {k: round(len(cmap & s) / len(s), 3) for k, s in SETS.items()}
        rows[d.name] = dict(file=f.name, files=len(list(d.glob("*.ttf"))), **cov)
        # specimen
        font = ImageFont.truetype(str(f), 44)
        if "[" in f.name:
            try:
                font.set_variation_by_axes([400])
            except Exception:
                pass
        langs = [l for l, k in (("ja", "kana"), ("ko", "hangul"), ("sc", "gb1"), ("tc", "big5")) if cov[k] > 0.9]
        if not langs:
            langs = [max((("ja", "kana"), ("ko", "hangul"), ("sc", "gb1"), ("tc", "big5")), key=lambda t: cov[t[1]])[0]]
        img = Image.new("L", (1250, 30 + 62 * len(langs)), 255)
        dr = ImageDraw.Draw(img)
        dr.text((6, 4), f"{d.name}  {f.name}  " + " ".join(f"{k}={v}" for k, v in cov.items()), fill=0,
                font=ImageFont.load_default(14))
        for i, l in enumerate(langs):
            dr.text((10, 26 + 62 * i), TEXT[l], font=font, fill=0)
        img.save(OUT / f"{d.name}.png")
        rows[d.name]["langs"] = langs
    (ROOT / "research" / "coverage.json").write_text(json.dumps(rows, indent=1))
    for k, r in rows.items():
        print(f"{k:26s} {r['files']:3d} {' '.join(f'{x}={r[x]:.2f}' for x in SETS)} {r['langs']}")


if __name__ == "__main__":
    main()
