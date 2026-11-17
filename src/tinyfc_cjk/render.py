"""Synthetic training data: random CJK text lines rendered with labelled fonts."""

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .taxonomy import COARSE, FONT_DIR, FONTS, LANGS, ROOT, Font, fine_label

H, W = 64, 256  # model input (grayscale)
RENDER_PX = 64
DATA = ROOT / "data"

HIRAGANA = [chr(c) for c in range(0x3041, 0x3094) if chr(c) not in "ゎゐゑ"]
KATAKANA = [chr(c) for c in range(0x30A1, 0x30F7) if chr(c) not in "ヮヰヱヵヶ"] + ["ー"]
PUNCT = {"ja": "、。「」？！・", "zh": "，。、「」？！", "ko": ".,?!"}
# Common Hangul syllables (frequent in running text) so lines are not all rare syllables.
HANGUL_COMMON = (
    "이다는의에가을를고하지기로서도한있어게요나사리아수것들면그주보거시대해자니라"
    "스만우일전부정무소중말여까오적그마때내안터비상제위다선연과생방인원성실관학"
    "국문화경세계사람우리하나모두한국어글세리프비교가반워사랑해요오늘날씨좋습니다"
)


def _decode_range(enc, rows, cols):
    out = []
    for b1 in rows:
        for b2 in cols:
            try:
                out.append(bytes([b1, b2]).decode(enc))
            except UnicodeDecodeError:
                pass
    return out


@lru_cache(None)
def pools():
    """Character pools per script: JIS X 0208 level 1, GB2312 level 1, Big5 common, KS X 1001 Hangul."""
    return dict(
        jis=_decode_range("euc_jp", range(0xB0, 0xD0), range(0xA1, 0xFF)),
        gb=_decode_range("gb2312", range(0xB0, 0xD8), range(0xA1, 0xFF)),
        big5=_decode_range("big5", range(0xA4, 0xC7), list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF))),
        hangul=_decode_range("euc_kr", range(0xB0, 0xC9), range(0xA1, 0xFF)),
    )


class Variant:
    """One face of a font at one weight. Variable fonts give several variants."""

    def __init__(self, font: Font, path: Path, wght=None, index=0):
        self.font, self.path, self.wght, self.index = font, path, wght, index

    @property
    def key(self):
        return str(self.path.relative_to(FONT_DIR))

    @property
    def name(self):
        return self.path.stem + (f"@{int(self.wght)}" if self.wght is not None else "")

    def pil(self, size=RENDER_PX):
        return _load_font(str(self.path), self.wght, self.index, size)


@lru_cache(maxsize=256)
def _load_font(path, wght, index, size):
    f = ImageFont.truetype(path, size, index=index)
    if wght is not None:
        f.set_variation_by_axes([wght])
    return f


def font_files(font: Font):
    files = sorted(p for p in font.path.iterdir() if p.suffix.lower() in (".ttf", ".otf", ".ttc"))
    return [p for p in files if "Italic" not in p.name and not any(x in p.name for x in font.exclude)]


def discover_variants(fonts=FONTS, var_weights=(300, 400, 500, 700, 900)):
    out = []
    for font in fonts:
        for p in font_files(font):
            if "[" in p.name:
                axes = {a.axisTag: a for a in TTFont(p, lazy=True)["fvar"].axes}
                lo, hi = axes["wght"].minValue, axes["wght"].maxValue
                ws = [w for w in var_weights if lo <= w <= hi] or [axes["wght"].defaultValue]
                out += [Variant(font, p, float(w)) for w in ws]
            else:
                out.append(Variant(font, p))
    return out


def _cmap(path: Path):
    if path.suffix.lower() == ".ttc":
        return set(TTCollection(path, lazy=True).fonts[0].getBestCmap())
    return set(TTFont(path, lazy=True).getBestCmap())


def charsets(fonts=FONTS, cache=DATA / "charsets.json"):
    """font file -> {script: chars}. Each face only draws characters it has, from fixed
    pools, so missing glyphs (tofu) never appear. Han pools below 90 % coverage are
    dropped (e.g. an SC-only face gets no Big5 pool)."""
    import json

    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    P = pools()
    out = {}
    for font in fonts:
        for path in font_files(font):
            cm = _cmap(path)
            has = lambda chars: "".join(c for c in chars if ord(c) in cm)
            d = {}
            if "ja" in font.langs:
                d["kana"] = has(HIRAGANA + KATAKANA)
                d["jis"] = has(P["jis"])
            if "zh" in font.langs or "ja" in font.langs:
                for k in ("gb", "big5"):
                    s = has(P[k])
                    if len(s) > 0.9 * len(P[k]):
                        d[k] = s
            if "ko" in font.langs:
                d["hangul"] = has(P["hangul"])
            for lang in font.langs:
                d["punct_" + lang] = has(PUNCT[lang])
            if "zh" in font.langs and not ({"gb", "big5", "jis"} & set(d)):
                raise ValueError(f"{path.name}: no Han pool with 90 % coverage")
            out[str(path.relative_to(FONT_DIR))] = d
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def sample_text(cs: dict, lang: str, rng: random.Random, n=None):
    """Random line for a script. lang: 'ja' kana-mixed, 'zh' Han only, 'ko' Hangul."""
    n = n or rng.randint(3, 9)
    punct = cs.get("punct_" + lang) or ""
    out = []
    if lang == "ja":
        kana = cs["kana"]
        hira = [c for c in kana if c <= "ゖ"] or kana
        kata = [c for c in kana if c > "ゖ"] or kana
        for _ in range(n):
            r = rng.random()
            pool = hira if r < 0.4 else kata if r < 0.62 else cs["jis"] if r < 0.97 or not punct else punct
            out.append(rng.choice(pool))
    elif lang == "zh":
        han = cs.get(rng.choice([k for k in ("gb", "big5") if k in cs])) if ("gb" in cs or "big5" in cs) else cs["jis"]
        for _ in range(n):
            out.append(rng.choice(punct) if punct and rng.random() < 0.04 else rng.choice(han))
    else:
        common = [c for c in HANGUL_COMMON if c in cs["hangul"]]
        for _ in range(n):
            r = rng.random()
            if r < 0.12:
                out.append(" ")
            elif r < 0.15 and punct:
                out.append(rng.choice(punct))
            else:
                out.append(rng.choice(common if r < 0.55 else cs["hangul"]))
        if out[0] == " ":
            out[0] = rng.choice(cs["hangul"])
    return "".join(out)


def render_line(variant: Variant, text: str, tracking=0.0) -> Image.Image:
    """Dark-on-light 'L' image of the text, cropped to the ink."""
    font = variant.pil()
    pad = RENDER_PX
    width = int(len(text) * RENDER_PX * (1.3 + max(tracking, 0)) + 2 * pad)
    img = Image.new("L", (width, RENDER_PX * 3), 255)
    d = ImageDraw.Draw(img)
    x, y = pad, int(RENDER_PX * 2)
    for ch in text:
        try:
            d.text((x, y), ch, font=font, fill=0, anchor="ls")
        except OSError:  # a few old fonts have glyphs with broken hinting bytecode
            continue
        x += font.getlength(ch) + tracking * RENDER_PX
    bbox = Image.eval(img, lambda p: 255 - p).getbbox()
    return img.crop(bbox) if bbox else img


def fit_to_canvas(line: Image.Image, rng: random.Random | None, text_h=None) -> np.ndarray:
    """Scale an ink-cropped line to height text_h and place it on an H x W canvas."""
    text_h = text_h or int(H * 0.8)
    s = text_h / max(line.height, 1)
    nw = max(1, round(line.width * s))
    resample = Image.Resampling.BILINEAR if rng and rng.random() < 0.5 else Image.Resampling.LANCZOS
    line = line.resize((nw, text_h), resample)
    canvas = Image.new("L", (W, H), 255)
    if nw > W:
        x0 = rng.randint(0, nw - W) if rng else 0
        line = line.crop((x0, 0, x0 + W, text_h))
        nw = W
    ox = rng.randint(0, W - nw) if rng else (W - nw) // 2
    oy = rng.randint(0, H - text_h) if rng else (H - text_h) // 2
    canvas.paste(line, (ox, oy))
    return np.asarray(canvas)


def augment(line: Image.Image, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.3:  # aspect change
        sx = rng.uniform(0.85, 1.15)
        line = line.resize((max(1, int(line.width * sx)), line.height))
    arr = fit_to_canvas(line, rng, text_h=rng.randint(28, 60))
    img = Image.fromarray(arr)
    if rng.random() < 0.3:  # low resolution
        f = rng.uniform(0.35, 0.8)
        img = img.resize((int(W * f), int(H * f)), Image.Resampling.BILINEAR).resize((W, H), Image.Resampling.BILINEAR)
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    a = np.asarray(img, dtype=np.float32) / 255.0
    fg, bg = rng.uniform(0.0, 0.35), rng.uniform(0.7, 1.0)
    a = bg + (fg - bg) * (1.0 - a)
    if rng.random() < 0.5:
        a = a + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(0, 0.05), a.shape)
    return np.clip(a * 255, 0, 255).astype(np.uint8)


@lru_cache(None)
def _kana_ratio_table():
    import json

    p = DATA / "kana_ratio_design.json"
    return json.loads(p.read_text()) if p.exists() else {}


def kana_ratio(variant: Variant) -> float:
    """The design's kana/kanji area ratio at Regular weight (NaN if unknown)."""
    t = _kana_ratio_table()
    return float(t.get("_per_file", {}).get(variant.key, t.get(variant.font.id, float("nan"))))


def make_sample(variant: Variant, lang: str, cs: dict, rng: random.Random):
    """-> (image uint8 HxW, coarse, script, fine_head, fine_cls, kana_ratio).
    Fonts listed for Japanese also give Han-only lines 15 % of the time; those count
    as script 'zh' and carry no Japanese fine label (the old/modern cue is in the kana)."""
    script = lang
    if lang == "ja" and rng.random() < 0.15:
        script = "zh"
    text = sample_text(cs, script, rng)
    tracking = rng.choice([0.0, 0.0, rng.uniform(-0.05, 0.25)])
    img = augment(render_line(variant, text, tracking), rng)
    head, cls = fine_label(variant.font, lang) if script == lang else (-1, -1)
    ratio = kana_ratio(variant) if script == "ja" else float("nan")
    return img, COARSE.index(variant.font.coarse), LANGS.index(script), head, cls, ratio
