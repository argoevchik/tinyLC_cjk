"""Measure the article's old-style vs modern Gothic traits on candidate Japanese sans fonts.

Article criteria: old-style = small, delicate counters and small letter (kana) size,
some brush movement; modern = larger counters/letter size; geometric/systematic =
prominently larger counters and letter size, systematic strokes.

Measured at 400 weight (or the nearest static weight), 200 px em:
  kana_h   mean ink height of kana / em        (letter size of kana)
  kana_a   mean ink-bbox area of kana / mean ink-bbox area of kanji
  counter  white fraction inside the ink bbox of 口日目田回 (counter size)
  ink      ink fraction of kanji bbox (stroke weight, to read counters against)
Also writes a side-by-side sheet of the article's figure-5 text.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EM = 200
KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん" \
       "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
KANJI = "書体語国東京明朝漢字永中新聞会社時間電話道場様愛"
COUNTER = "口日目田回"


def font(path, wght=None, index=0):
    f = ImageFont.truetype(str(path), EM, index=index)
    if wght is not None:
        try:
            f.set_variation_by_axes([wght])
        except Exception:
            pass
    return f


def glyph(f, ch):
    img = Image.new("L", (EM * 2, EM * 2), 0)
    try:
        ImageDraw.Draw(img).text((EM // 2, EM // 2), ch, font=f, fill=255)
    except OSError:  # broken hinting bytecode in a few old fonts
        return None
    a = np.asarray(img) > 127
    ys, xs = np.where(a)
    if not len(xs):
        return None
    return a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def traits(f):
    kana = [g for g in (glyph(f, c) for c in KANA) if g is not None]
    kanji = [g for g in (glyph(f, c) for c in KANJI) if g is not None]
    cnt = [g for g in (glyph(f, c) for c in COUNTER) if g is not None]
    area = lambda gs: np.mean([g.shape[0] * g.shape[1] for g in gs])
    return dict(
        kana_h=round(float(np.mean([g.shape[0] for g in kana])) / EM, 3),
        kana_a=round(float(area(kana) / area(kanji)), 3),
        counter=round(float(np.mean([1 - g.mean() for g in cnt])), 3),
        ink=round(float(np.mean([g.mean() for g in kanji])), 3),
    )


def main(cands):
    rows, sheet = {}, []
    for name, path, wght, idx in cands:
        f = font(path, wght, idx)
        rows[name] = traits(f)
        print(f"{name:26s} " + " ".join(f"{k}={v:.3f}" for k, v in rows[name].items()))
        im = Image.new("L", (1500, 120), 255)
        d = ImageDraw.Draw(im)
        small = ImageFont.truetype(str(path), 84, index=idx)
        if wght is not None:
            try:
                small.set_variation_by_axes([wght])
            except Exception:
                pass
        d.text((260, 10), "オールド書体はどんな？", font=small, fill=0)
        d.text((4, 50), name, font=ImageFont.load_default(20), fill=0)
        sheet.append(im)
    out = ROOT / "research" / "jp_gothic_traits.json"
    out.write_text(json.dumps(rows, indent=1))
    s = Image.new("L", (1500, 120 * len(sheet)), 255)
    for i, im in enumerate(sheet):
        s.paste(im, (0, 120 * i))
    s.save(ROOT / "research" / "jp_gothic_sheet.png")


if __name__ == "__main__":
    F, D = ROOT / "fonts", ROOT / "ext" / "deb"
    cands = [
        ("zenkakugothicnew*", F / "zenkakugothicnew/ZenKakuGothicNew-Regular.ttf", None, 0),
        ("zenkakugothicantique", F / "zenkakugothicantique/ZenKakuGothicAntique-Regular.ttf", None, 0),
        ("shipporiantique", F / "shipporiantique/ShipporiAntique-Regular.ttf", None, 0),
        ("notosansjp*", F / "notosansjp/NotoSansJP[wght].ttf", 400, 0),
        ("ibmplexsansjp", F / "ibmplexsansjp/IBMPlexSansJP-Regular.ttf", None, 0),
        ("bizudpgothic", F / "bizudpgothic/BIZUDPGothic-Regular.ttf", None, 0),
        ("murecho", F / "murecho/Murecho[wght].ttf", 400, 0),
        ("sawarabigothic", F / "sawarabigothic/SawarabiGothic-Regular.ttf", None, 0),
        ("mplus1p*", F / "mplus1p/MPLUS1p-Regular.ttf", None, 0),
        ("mplus2", F / "mplus2/MPLUS2[wght].ttf", 400, 0),
        ("lineseedjp", F / "lineseedjp/LINESeedJP-Regular.ttf", None, 0),
        ("kosugi", F / "kosugi/Kosugi-Regular.ttf", None, 0),
        ("wdxllubrifontjpn", F / "wdxllubrifontjpn/WDXLLubrifontJPN-Regular.ttf", None, 0),
        ("chocolateclassicalsans", F / "chocolateclassicalsans/ChocolateClassicalSans-Regular.ttf", None, 0),
        ("lxgwmarkergothic", F / "lxgwmarkergothic/LXGWMarkerGothic-Regular.ttf", None, 0),
        ("delagothicone", F / "delagothicone/DelaGothicOne-Regular.ttf", None, 0),
        ("rocknrollone", F / "rocknrollone/RocknRollOne-Regular.ttf", None, 0),
        ("notosanssc(jp kana)", F / "notosanssc/NotoSansSC[wght].ttf", 400, 0),
        ("ipaexgothic", D / "fonts-ipaexfont-gothic/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf", None, 0),
        ("ume-tgo5", D / "fonts-horai-umefont/usr/share/fonts/truetype/horai-umefont/ume-tgo5.ttf", None, 0),
        ("ume-tgc4", D / "fonts-horai-umefont/usr/share/fonts/truetype/horai-umefont/ume-tgc4.ttf", None, 0),
        ("ume-tgs4", D / "fonts-horai-umefont/usr/share/fonts/truetype/horai-umefont/ume-tgs4.ttf", None, 0),
        ("vlgothic", D / "fonts-vlgothic/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf", None, 0),
        ("konatu", D / "fonts-konatu/usr/share/fonts/truetype/konatu/Konatu.ttf", None, 0),
        ("wqy-zenhei", D / "fonts-wqy-zenhei/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", None, 0),
        ("wqy-microhei", D / "fonts-wqy-microhei/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", None, 0),
    ]
    main(cands)
