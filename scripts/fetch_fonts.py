"""Download every font the taxonomy uses, with its licence, into fonts/.

  ./run.sh scripts/fetch_fonts.py                 # all fonts (about 1.4 GB)
  ./run.sh scripts/fetch_fonts.py --only kosugi konatu
  ./run.sh scripts/fetch_fonts.py --dest /tmp/f   # somewhere else

* Google Fonts families come from github.com/google/fonts (main branch), one sparse
  checkout of the needed folders: font files plus OFL.txt / LICENSE.txt / METADATA.pb.
  They land in fonts/<id>/.
* The other designs come from Debian `main` packages (deb.debian.org, newest version
  in the pool). The .deb is unpacked here, so neither `ar` nor `dpkg` is needed. Only
  the listed font files are kept, plus the package's copyright file as
  LICENSE-<package>.txt. They land in fonts/_extra/<id>/.
Existing folders that already hold font files are skipped unless --force is given.
Needs `git` and network access. Newer upstream versions can differ slightly from
the ones the published model was trained on (2026-11-17).
"""

import argparse
import fnmatch
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tinyfc_cjk.taxonomy import FONTS  # noqa: E402

DEBIAN = "http://deb.debian.org/debian/pool/main"
GF_REPO = "https://github.com/google/fonts.git"
FONT_EXT = (".ttf", ".otf", ".ttc")

# id -> [(source package dir in the pool, binary package, member globs)]
EXTRA = {
    "ipaexgothic": [("fonts-ipaexfont", "fonts-ipaexfont-gothic", ["*/ipaexg.ttf"])],
    "ipaexmincho": [("fonts-ipaexfont", "fonts-ipaexfont-mincho", ["*/ipaexm.ttf"])],
    "konatu": [("fonts-konatu", "fonts-konatu", ["*.ttf"])],
    "vlgothic": [("fonts-vlgothic", "fonts-vlgothic", ["*.ttf"])],
    "umegothic": [("fonts-horai-umefont", "fonts-horai-umefont", ["*/ume-[tpuh]g*.ttf"])],
    "umemincho": [("fonts-horai-umefont", "fonts-horai-umefont", ["*/ume-[tp]m*.ttf"])],
    "hanazono": [("fonts-hanazono", "fonts-hanazono", ["*/HanaMinA.ttf"])],
    "kouzan": [("fonts-kouzan-mouhitsu", "fonts-kouzan-mouhitsu", ["*.ttf"]),
               ("fonts-aoyagi-kouzan-t", "fonts-aoyagi-kouzan-t", ["*.ttf"])],
    "aoyagisoseki": [("fonts-aoyagi-soseki", "fonts-aoyagi-soseki", ["*.ttf"])],
    "arphicming": [("fonts-arphic-uming", "fonts-arphic-uming", ["*/uming.ttc"]),
                   ("fonts-arphic-bsmi00lp", "fonts-arphic-bsmi00lp", ["*.ttf"]),
                   ("fonts-arphic-gbsn00lp", "fonts-arphic-gbsn00lp", ["*.ttf"])],
    "arphickai": [("fonts-arphic-ukai", "fonts-arphic-ukai", ["*/ukai.ttc"]),
                  ("fonts-arphic-gkai00mp", "fonts-arphic-gkai00mp", ["*.ttf"]),
                  ("fonts-arphic-bkai00mp", "fonts-arphic-bkai00mp", ["*.ttf"])],
    "wqyzenhei": [("fonts-wqy-zenhei", "fonts-wqy-zenhei", ["*/wqy-zenhei.ttc"])],
    "wqymicrohei": [("fonts-wqy-microhei", "fonts-wqy-microhei", ["*/wqy-microhei.ttc"])],
    "nanumbarungothic": [("fonts-nanum", "fonts-nanum", ["*/NanumBarunGothic.ttf", "*/NanumBarunGothicBold.ttf"])],
    "nanumsquare": [("fonts-nanum", "fonts-nanum", ["*/NanumSquareR.ttf", "*/NanumSquareB.ttf",
                                                    "*/NanumSquareNeo-bRg.ttf", "*/NanumSquareNeo-cBd.ttf"])],
    "nanumsquareround": [("fonts-nanum", "fonts-nanum", ["*/NanumSquareRound*.ttf"])],
    "baekmukbatang": [("fonts-baekmuk", "fonts-baekmuk", ["*/batang.ttf"])],
    "baekmukdotum": [("fonts-baekmuk", "fonts-baekmuk", ["*/dotum.ttf", "*/gulim.ttf"])],
    "unbatang": [("fonts-unfonts-core", "fonts-unfonts-core", ["*/UnBatang*.ttf"])],
    "undotum": [("fonts-unfonts-core", "fonts-unfonts-core", ["*/UnDotum*.ttf"])],
    "ungraphic": [("fonts-unfonts-core", "fonts-unfonts-core", ["*/UnGraphic*.ttf"])],
    "ungungseo": [("fonts-unfonts-core", "fonts-unfonts-core", ["*/UnGungseo.ttf"])],
}


def has_fonts(d: Path):
    return d.is_dir() and any(p.suffix.lower() in FONT_EXT for p in d.iterdir())


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tinyfc-cjk-fetch-fonts"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ---------------------------------------------------------------- Google Fonts
def google_fonts(ids, dest: Path):
    if not ids:
        return
    print(f"google/fonts: {len(ids)} families (sparse checkout)")
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "gf"
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", GF_REPO, str(repo)],
                       check=True)
        patterns = [f"/{lic}/{i}/" for i in ids for lic in ("ofl", "apache", "ufl")]
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "--no-cone", *patterns], check=True)
        for i in ids:
            src = next((repo / lic / i for lic in ("ofl", "apache", "ufl") if (repo / lic / i).is_dir()), None)
            if src is None:
                print(f"  !! {i}: not found in google/fonts")
                continue
            out = dest / i
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            for p in src.iterdir():
                if p.is_file():
                    shutil.copy2(p, out / p.name)
            n = sum(1 for p in out.iterdir() if p.suffix.lower() in FONT_EXT)
            print(f"  {i}: {n} font files ({src.parent.name})")


# ---------------------------------------------------------------- Debian
_deb_cache: dict[str, bytes] = {}


def deb_url(pool_dir, package):
    listing = fetch(f"{DEBIAN}/{pool_dir[0]}/{pool_dir}/").decode("utf-8", "replace")
    names = sorted(set(re.findall(rf'href="({re.escape(package)}_[^"/]+_all\.deb)"', listing)),
                   key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)])
    if not names:
        raise RuntimeError(f"no {package}_*_all.deb in {DEBIAN}/{pool_dir[0]}/{pool_dir}/")
    return f"{DEBIAN}/{pool_dir[0]}/{pool_dir}/{names[-1]}"


def deb_data(pool_dir, package) -> tarfile.TarFile:
    """The data.tar.* member of the newest .deb, opened as a tar archive."""
    if package not in _deb_cache:
        url = deb_url(pool_dir, package)
        print(f"  download {url.rsplit('/', 1)[1]}")
        _deb_cache[package] = fetch(url)
    raw = _deb_cache[package]
    assert raw[:8] == b"!<arch>\n", f"{package}: not an ar archive"
    pos = 8
    while pos < len(raw):
        name = raw[pos:pos + 16].decode().strip().rstrip("/")
        size = int(raw[pos + 48:pos + 58].decode().strip())
        body = raw[pos + 60:pos + 60 + size]
        if name.startswith("data.tar"):
            if name.endswith(".zst"):
                body = subprocess.run(["zstd", "-dc"], input=body, capture_output=True, check=True).stdout
            return tarfile.open(fileobj=io.BytesIO(body), mode="r:*")
        pos += 60 + size + (size & 1)
    raise RuntimeError(f"{package}: no data.tar member")


def debian(ids, dest: Path):
    for i in ids:
        out = dest / "_extra" / i
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        n = 0
        for pool_dir, package, globs in EXTRA[i]:
            tar = deb_data(pool_dir, package)
            for m in tar.getmembers():
                if m.isfile() and any(fnmatch.fnmatch(m.name, g) for g in globs) and m.name.lower().endswith(FONT_EXT):
                    (out / Path(m.name).name).write_bytes(tar.extractfile(m).read())
                    n += 1
            lic = next((m for m in tar.getmembers() if m.name.endswith(f"usr/share/doc/{package}/copyright")), None)
            if lic is not None and lic.isfile():
                (out / f"LICENSE-{package}.txt").write_bytes(tar.extractfile(lic).read())
            else:
                print(f"  !! {package}: no copyright file in the package (see Debian's page for it)")
        print(f"  {i}: {n} font files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "fonts"))
    ap.add_argument("--only", nargs="*", help="font ids from taxonomy.py")
    ap.add_argument("--force", action="store_true", help="re-download folders that already have fonts")
    a = ap.parse_args()
    dest = Path(a.dest)
    ids = [f.id for f in FONTS]
    unknown = set(a.only or []) - set(ids)
    if unknown:
        sys.exit(f"unknown font ids: {sorted(unknown)}")
    if a.only:
        ids = [i for i in ids if i in a.only]
    todo = [i for i in ids if a.force or not has_fonts(dest / ("_extra/" + i if i in EXTRA else i))]
    print(f"{len(ids)} fonts requested, {len(ids) - len(todo)} already present, fetching {len(todo)}")
    google_fonts([i for i in todo if i not in EXTRA], dest)
    debian([i for i in todo if i in EXTRA], dest)
    missing = [i for i in ids if not has_fonts(dest / ("_extra/" + i if i in EXTRA else i))]
    print("done." if not missing else f"missing: {missing}")


if __name__ == "__main__":
    main()
