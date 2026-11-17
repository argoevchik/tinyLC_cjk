"""CJK type classification after Google Fonts Knowledge, "Type classification in CJK"
(Chinese, Japanese and Korean lessons).

  https://fonts.google.com/knowledge/type_in_china_japan_and_korea/type_classification_in_cjk_chinese
  https://fonts.google.com/knowledge/type_in_china_japan_and_korea/type_classification_in_cjk_japanese
  https://fonts.google.com/knowledge/type_in_china_japan_and_korea/type_classification_in_cjk_korean

Two levels:

* COARSE: four classes that exist in all three languages, shared by one model head.
    serif   Songti/Mingti 宋体 · Mincho 明朝 · Myeongjo/Batang 명조/바탕
    sans    Heiti 黑体 · Gothic ゴシック · Dotum/Gothic 돋움/고딕
    brush   Kaiti 楷体 and calligraphic/handwritten brush 行书 · 教科書体/毛筆 · 궁서/붓글씨
    rounded Yuanti 圆体 · 丸ゴシック · 둥근 고딕 (the Chinese lesson calls Yuanti a
            sub-group of Heiti; it is kept apart because it is visually distinct)

* FINE: per-language subclasses inside one coarse class. A fine head is trained
  only when **every** subclass has at least MIN_FINE_GROUPS distinct designs
  (design groups, see below). Candidate heads that fall short are listed in
  FINE_CANDIDATES but switched off; `enabled_fine_heads()` decides.

Each font entry has a design `group`. Fonts that share glyph outlines share a group,
across languages too (e.g. Noto Sans SC/JP/KR, Chiron Hei HK and Chocolate Classical
Sans are all Source Han Sans). Cross-validation holds out whole groups.

`source`: "article" = the lessons name or show the font as an example of the class;
"judgment" = my label, made against the lessons' criteria. Japanese and Korean
fine labels are backed by measurements (research/jp_gothic_traits.py, README):
kana/kanji ink-box area ratio for old-style vs modern Japanese, and Hangul advance
width (full-square vs 85-97 %) for Korean.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "fonts"

COARSE = ["serif", "sans", "brush", "rounded"]
LANGS = ["ja", "zh", "ko"]  # script of the rendered line: kana-mixed, Han only, Hangul
MIN_FINE_GROUPS = 4

COARSE_INFO = {
    "serif": dict(zh="宋体 Songti / 明体", ja="明朝体 Mincho", ko="명조/바탕 Myeongjo/Batang",
                  traits="Woodblock-derived: thin horizontals, thick verticals, triangular "
                         "serifs (uroko) at stroke ends, accentuated contrast."),
    "sans": dict(zh="黑体 Heiti", ja="ゴシック体 Gothic", ko="고딕/돋움 Gothic/Dotum",
                 traits="Even stroke weight, no serifs, square stroke ends."),
    "brush": dict(zh="楷体 Kaiti / 手写体·行书", ja="楷書体・教科書体・毛筆", ko="궁서/붓글씨",
                  traits="Brush movement kept: tapered, slanted, modulated strokes from "
                         "calligraphy (regular script Kai, or faster running/cursive hands)."),
    "rounded": dict(zh="圆体 Yuanti", ja="丸ゴシック Maru Gothic", ko="둥근 고딕 rounded",
                    traits="Sans without serifs whose stroke ends and corners are rounded."),
}

# (lang, coarse) -> ordered fine classes
FINE_CANDIDATES = {
    ("ja", "serif"): ["mincho_old", "mincho_modern"],
    ("ja", "sans"): ["gothic_old", "gothic_humanist", "gothic_geometric"],
    ("ko", "serif"): ["batang_old", "batang_modern"],
    ("ko", "sans"): ["dotum_old_square", "dotum_modern"],
    ("zh", "brush"): ["kaiti", "xingcao"],
    ("zh", "serif"): ["songti", "fangsong"],
}

FINE_INFO = {
    "mincho_old": "オールドスタイル明朝: small delicate counters, small kana, lingering brush strokes, organic curves.",
    "mincho_modern": "モダン明朝: larger, dynamic counters; simpler, clean strokes; sometimes low contrast.",
    "gothic_old": "オールドスタイルゴシック: old-style proportions in a sans (small kana and counters).",
    "gothic_humanist": "モダンゴシック (humanist): modern strokes, traditional letterforms.",
    "gothic_geometric": "モダンゴシック (systematic): prominently larger counters and kana, systematic strokes.",
    "batang_old": "Old-style Korean serif: small counters, brush-like lingering strokes, geometric Hunminjeongeum letterform.",
    "batang_modern": "Modern Korean serif: humanist or stiff text serif, larger letter size (Noto Serif KR, Nanum Myeongjo).",
    "dotum_old_square": "Traditional Korean sans: full-width square letterform, components stretched to fill it.",
    "dotum_modern": "Modern Korean sans: narrow width (85-97 %), components sized to respect Hangeul forms.",
    "kaiti": "楷体: standardized regular-script brush style.",
    "xingcao": "手写体/行书/草书: faster running or cursive hands.",
    "songti": "宋体: woodblock style with diamond-shaped serifs.",
    "fangsong": "仿宋体: Songti with slanted horizontals and sharper endings.",
}


@dataclass(frozen=True)
class Font:
    id: str
    coarse: str
    group: str
    langs: tuple  # which line scripts to render with this font
    fine: dict = field(default_factory=dict, hash=False)  # lang -> fine class
    source: str = "judgment"
    dir: str | None = None  # folder under fonts/ (default: id, or _extra/id)
    exclude: tuple = ()  # filename substrings to skip

    @property
    def path(self) -> Path:
        d = FONT_DIR / (self.dir or self.id)
        return d if d.exists() else FONT_DIR / "_extra" / self.id


def F(id, coarse, group=None, langs=("ja",), fine=None, source="judgment", **kw):
    if isinstance(fine, str):
        fine = {langs[0]: fine}
    return Font(id, coarse, group or id, tuple(langs), fine or {}, source, **kw)


JA, ZH, KO = ("ja",), ("zh",), ("ko",)
FONTS = [
    # ---------------- Japanese serif (Mincho)
    F("zenoldmincho", "serif", fine="mincho_old", source="article"),
    F("shipporimincho", "serif", fine="mincho_old", source="article"),
    F("shipporiminchob1", "serif", "shipporimincho", fine="mincho_old"),
    F("hinamincho", "serif", fine="mincho_old"),
    F("kaiseiopti", "serif", "kaisei", fine="mincho_old"),
    F("kaiseitokumin", "serif", "kaisei", fine="mincho_old"),
    F("kaiseiharunoumi", "serif", "kaisei", fine="mincho_old"),
    F("newtegomin", "serif", fine="mincho_old"),
    F("zenantique", "serif", fine="mincho_old"),
    F("notoserifjp", "serif", "sourcehanserif", fine="mincho_modern", source="article"),
    F("bizudmincho", "serif", "bizudmincho", fine="mincho_modern"),
    F("bizudpmincho", "serif", "bizudmincho", fine="mincho_modern"),
    F("sawarabimincho", "serif", fine="mincho_modern"),
    F("ipaexmincho", "serif", fine="mincho_modern"),
    F("hanazono", "serif", fine="mincho_modern"),
    F("umemincho", "serif", "ume", fine="mincho_modern"),
    F("cactusclassicalserif", "serif", "cactus", langs=("ja", "zh"), fine={"ja": "mincho_modern"}),
    # ---------------- Japanese sans (Gothic)
    F("zenkakugothicnew", "sans", "zenkakugothic", fine="gothic_old", source="article"),
    F("zenkakugothicantique", "sans", "zenkakugothic", fine="gothic_old"),
    F("shipporiantique", "sans", fine="gothic_old"),
    F("shipporiantiqueb1", "sans", "shipporiantique", fine="gothic_old"),
    F("konatu", "sans", fine="gothic_old"),
    F("ipaexgothic", "sans", fine="gothic_old"),
    F("notosansjp", "sans", "sourcehansans", fine="gothic_humanist", source="article"),
    F("ibmplexsansjp", "sans", "ibmplex", fine="gothic_humanist"),
    F("bizudgothic", "sans", "bizudgothic", fine="gothic_humanist"),
    F("bizudpgothic", "sans", "bizudgothic", fine="gothic_humanist"),
    F("sawarabigothic", "sans", fine="gothic_humanist"),
    F("mplus1p", "sans", "mplus", fine="gothic_geometric", source="article"),
    F("mplus1", "sans", "mplus", fine="gothic_geometric"),
    F("mplus2", "sans", "mplus", fine="gothic_geometric"),
    F("mplus1code", "sans", "mplus", fine="gothic_geometric"),
    F("mplusu", "sans", "mplus", fine="gothic_geometric"),
    F("vlgothic", "sans", "mplus", fine="gothic_geometric"),  # kana measure identical to M+
    F("lineseedjp", "sans", fine="gothic_geometric"),
    F("murecho", "sans", fine="gothic_geometric"),
    F("kosugi", "sans", "motoya", fine="gothic_geometric"),
    F("umegothic", "sans", "ume", fine="gothic_geometric"),
    F("wqymicrohei", "sans", "droidsansfallback", langs=("ja", "zh"), fine={"ja": "gothic_geometric"}),
    # ---------------- Japanese rounded / brush
    F("kosugimaru", "rounded", "motoya"),
    F("zenmarugothic", "rounded"),
    F("mplusrounded1c", "rounded", "mplus"),
    F("kiwimaru", "rounded"),
    F("mochiypopone", "rounded", "mochiy"),
    F("mochiypoppone", "rounded", "mochiy"),
    F("kleeone", "brush", "klee"),
    F("yujisyuku", "brush", "yuji"),
    F("yujiboku", "brush", "yuji"),
    F("yujimai", "brush", "yuji"),
    F("kouzan", "brush"),
    F("aoyagisoseki", "brush"),
    F("zenkurenaido", "brush"),
    F("pottaone", "brush"),
    # ---------------- Chinese
    F("notoserifsc", "serif", "sourcehanserif", ZH, fine={"zh": "songti"}, source="article"),
    F("notoseriftc", "serif", "sourcehanserif", ZH, fine={"zh": "songti"}),
    F("notoserifhk", "serif", "sourcehanserif", ZH, fine={"zh": "songti"}),
    F("chironsunghk", "serif", "sourcehanserif", ZH, fine={"zh": "songti"}),
    F("arphicming", "serif", langs=ZH, fine={"zh": "songti"}),
    F("uoqmunthenkhung", "serif", langs=ZH, fine={"zh": "songti"}),
    F("zcoolxiaowei", "serif", langs=ZH, fine={"zh": "songti"}),
    F("notosanssc", "sans", "sourcehansans", ZH, source="article"),
    F("notosanstc", "sans", "sourcehansans", ZH),
    F("notosanshk", "sans", "sourcehansans", ZH),
    F("chironheihk", "sans", "sourcehansans", ZH),
    F("chocolateclassicalsans", "sans", "sourcehansans", ZH),
    F("wqyzenhei", "sans", langs=ZH),
    F("wdxllubrifontsc", "sans", "wdxl", ZH),
    F("wdxllubrifonttc", "sans", "wdxl", ZH),
    F("lxgwmarkergothic", "sans", langs=ZH),
    F("mashanzheng", "brush", langs=ZH, fine="kaiti"),
    F("lxgwwenkaitc", "brush", "klee", ZH, fine="kaiti"),
    F("lxgwwenkaimonotc", "brush", "klee", ZH, fine="kaiti"),
    F("iansui", "brush", "klee", ZH, fine="kaiti"),
    F("arphickai", "brush", langs=ZH, fine="kaiti"),
    F("zhimangxing", "brush", langs=ZH, fine="xingcao"),
    F("longcang", "brush", langs=ZH, fine="xingcao"),
    F("liujianmaocao", "brush", langs=ZH, fine="xingcao", source="article"),
    F("huninn", "rounded", "motoya", ZH),  # jf open-huninn is derived from Kosugi Maru
    F("chirongoroundtc", "rounded", "sourcehansans", ("zh", "ja", "ko")),
    # ---------------- Korean
    F("gowunbatang", "serif", langs=KO, fine="batang_old", source="article"),
    F("songmyung", "serif", langs=KO, fine="batang_old"),
    F("unbatang", "serif", langs=KO, fine="batang_old"),
    F("baekmukbatang", "serif", langs=KO, fine="batang_old"),
    F("notoserifkr", "serif", "sourcehanserif", KO, fine="batang_modern", source="article"),
    F("nanummyeongjo", "serif", langs=KO, fine="batang_modern", source="article"),
    F("hahmlet", "serif", langs=KO, fine="batang_modern"),
    F("diphylleia", "serif", langs=KO, fine="batang_modern"),
    F("notosanskr", "sans", "sourcehansans", KO, fine="dotum_modern", source="article"),
    F("ibmplexsanskr", "sans", "ibmplex", KO, fine="dotum_modern", source="article"),
    F("nanumgothic", "sans", langs=KO, fine="dotum_modern"),
    F("nanumgothiccoding", "sans", "nanumgothic", KO, fine="dotum_modern"),
    F("nanumbarungothic", "sans", langs=KO, fine="dotum_modern"),
    F("gothica1", "sans", langs=KO, fine="dotum_modern"),
    F("astasans", "sans", langs=KO, fine="dotum_modern"),
    F("orbit", "sans", langs=KO, fine="dotum_modern"),
    F("gowundodum", "sans", langs=KO, fine="dotum_modern"),
    F("sunflower", "sans", langs=KO, fine="dotum_modern"),
    F("nanumsquare", "sans", langs=KO, fine="dotum_modern"),
    F("baekmukdotum", "sans", "baekmuk", KO, fine="dotum_old_square"),
    F("undotum", "sans", "un", KO, fine="dotum_old_square"),
    F("ungraphic", "sans", "ungraphic", KO, fine="dotum_old_square"),
    F("nanumbrushscript", "brush", langs=KO),
    F("eastseadokdo", "brush", langs=KO),
    F("dokdo", "brush", langs=KO),
    F("yeonsung", "brush", langs=KO),
    F("ungungseo", "brush", langs=KO),
    F("jua", "rounded", langs=KO),
    F("dongle", "rounded", langs=KO, source="article"),  # the lesson's Talnemo example
    F("nanumsquareround", "rounded", "nanumsquare", KO),
]


# Japanese old-style vs modern, measured: ink-box area of kana / of kanji at Regular
# weight (research/jp_gothic_traits.py, data/kana_ratio_design.json). The model also
# regresses this number; these cuts turn it into a fine class, as an alternative to
# the fine head. Cuts sit between the article's examples:
#   Mincho: Zen Old 0.59, Shippori 0.61 (old) | Noto Serif JP 0.67 (modern)
#   Gothic: Zen Kaku New 0.64 (old) | Noto Sans JP 0.75 (humanist) | M PLUS 1p 0.79 (geometric)
TRAIT_CUTS = {
    ("ja", "serif"): [0.652],
    ("ja", "sans"): [0.73, 0.762],
}


def trait_class(key, value):
    cuts = TRAIT_CUTS[key]
    return FINE_CANDIDATES[key][sum(value >= c for c in cuts)]


def fine_groups():
    """(lang, coarse) -> fine class -> set of design groups."""
    out = defaultdict(lambda: defaultdict(set))
    for f in FONTS:
        for lang, fc in f.fine.items():
            out[(lang, f.coarse)][fc].add(f.group)
    return out


def enabled_fine_heads(min_groups=MIN_FINE_GROUPS):
    fg = fine_groups()
    heads = {}
    for key, classes in FINE_CANDIDATES.items():
        if all(len(fg[key][c]) >= min_groups for c in classes):
            heads[key] = classes
    return heads


FINE_HEADS = enabled_fine_heads()
HEAD_KEYS = list(FINE_HEADS)  # index -> (lang, coarse)


def fine_label(font: Font, lang: str):
    """(head index, class index) or (-1, -1)."""
    key = (lang, font.coarse)
    if key in FINE_HEADS and lang in font.fine:
        return HEAD_KEYS.index(key), FINE_HEADS[key].index(font.fine[lang])
    return -1, -1


def cv_folds(n_folds=4):
    """Assign whole design groups to folds, spread so that every (coarse, lang)
    cell and every fine class loses only part of its groups in any fold."""
    cells = defaultdict(set)
    for f in FONTS:
        for lang in f.langs:
            key = (lang, f.coarse, f.fine.get(lang, ""))
            cells[key].add(f.group)
    fold_of, load = {}, [0] * n_folds
    # small cells first, so their groups get spread before big cells fill folds
    for key in sorted(cells, key=lambda k: (len(cells[k]), k)):
        taken = [fold_of[g] for g in cells[key] if g in fold_of]
        for g in sorted(cells[key]):
            if g in fold_of:
                continue
            k = min(range(n_folds), key=lambda i: (taken.count(i), load[i], i))
            fold_of[g] = k
            taken.append(k)
            load[k] += 1
    return [sorted(g for g, k in fold_of.items() if k == i) for i in range(n_folds)]


if __name__ == "__main__":
    fg = fine_groups()
    print("coarse groups per language:")
    for lang in LANGS:
        row = {c: sorted({f.group for f in FONTS if f.coarse == c and lang in f.langs}) for c in COARSE}
        print(f"  {lang}: " + "  ".join(f"{c}={len(g)}" for c, g in row.items()))
    print("\nfine candidates (need >=%d groups per class):" % MIN_FINE_GROUPS)
    for key, classes in FINE_CANDIDATES.items():
        state = "ON " if key in FINE_HEADS else "off"
        print(f"  [{state}] {key[0]}/{key[1]}: " + ", ".join(f"{c}={len(fg[key][c])} {sorted(fg[key][c])}" for c in classes))
    missing = [f.id for f in FONTS if not f.path.exists()]
    print("\nmissing font dirs:", missing)
    for i, fold in enumerate(cv_folds()):
        print(f"fold {i}: {fold}")
