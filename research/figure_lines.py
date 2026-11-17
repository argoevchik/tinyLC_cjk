"""Cut text lines out of the three lessons' figures -> research/figure_lines/*.png + index.

Bands of ink rows are lines; inside a band, segments separated by wide gaps are
columns; segments much shorter than the band (the small font-name captions) are dropped.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(__file__).resolve().parent
OUT = R / "figure_lines"
OUT.mkdir(exist_ok=True)


def runs(mask, min_gap):
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > min_gap:
            out.append((s, p + 1))
            s = i
        p = i
    out.append((s, p + 1))
    return out


def cut(png):
    a = np.asarray(Image.open(png).convert("RGB")).astype(int)
    ink = a.min(2) < 190
    if png.stem.startswith(("chinese", "japanese_01", "japanese_02", "korean_06")):
        ink[:, : int(0.235 * ink.shape[1])] = False  # font-name captions sit in this strip
    lines = []
    for y0, y1 in runs(ink.any(1), 6):
        if y1 - y0 < 45:  # captions
            continue
        band = ink[y0:y1]
        for x0, x1 in runs(band.any(0), 70):
            seg = band[:, x0:x1]
            ys = np.where(seg.any(1))[0]
            if ys[-1] - ys[0] < 0.55 * (y1 - y0) or x1 - x0 < 150:
                continue
            lines.append((int(x0), int(y0 + ys[0]), int(x1), int(y0 + ys[-1] + 1)))
    return lines


def main():
    idx = []
    for png in sorted((R / "figs").glob("*_*.png")):
        if "thumbnail" in png.name:
            continue
        img = Image.open(png).convert("RGB")
        for i, (x0, y0, x1, y1) in enumerate(cut(png)):
            name = f"{png.stem}_{i:02d}.png"
            img.crop((x0 - 4, y0 - 4, x1 + 4, y1 + 4)).save(OUT / name)
            idx.append(dict(file=name, box=[x0, y0, x1, y1]))
    (R / "figure_lines.json").write_text(json.dumps(idx, indent=1))
    print(len(idx), "lines")
    # contact sheet for checking
    ims = [Image.open(OUT / d["file"]) for d in idx]
    ims = [im.resize((max(1, im.width * 40 // im.height), 40)) for im in ims]
    sheet = Image.new("RGB", (1300, 46 * len(ims)), "white")
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for i, (d, im) in enumerate(zip(idx, ims)):
        dr.text((2, 46 * i + 14), d["file"].replace(".png", ""), fill="black")
        sheet.paste(im.crop((0, 0, min(im.width, 1000), 40)), (290, 46 * i))
    sheet.save(R / "figure_lines_sheet.png")


if __name__ == "__main__":
    main()
