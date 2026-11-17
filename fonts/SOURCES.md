# Font sources

Fonts are not in the repository: `./run.sh scripts/fetch_fonts.py` downloads them,
each with its licence file.

* `fonts/<id>/` — Google Fonts families from [google/fonts](https://github.com/google/fonts)
  (`ofl/`, `apache/`, main branch): font files plus `OFL.txt` / `LICENSE.txt` and `METADATA.pb`.
  OFL 1.1 / Apache 2.0. The published model was trained on the versions of 2026-11-17.
* `fonts/_extra/<id>/` — designs Google Fonts lacks, from Debian `main` packages
  (deb.debian.org, newest version in the pool), with the package's copyright file as
  `LICENSE-<package>.txt`:

| id | Debian package | licence |
|---|---|---|
| ipaexgothic, ipaexmincho | fonts-ipaexfont-gothic / -mincho | IPA Font License |
| konatu | fonts-konatu | free (author's terms, see package) |
| vlgothic | fonts-vlgothic | M+ licence + BSD |
| umegothic, umemincho | fonts-horai-umefont | Ume font licence (free) |
| hanazono | fonts-hanazono | free (Hanazono licence) |
| kouzan | fonts-kouzan-mouhitsu, fonts-aoyagi-kouzan-t | free (author) / OFL |
| aoyagisoseki | fonts-aoyagi-soseki | free (author) |
| arphicming, arphickai | fonts-arphic-uming/-bsmi00lp/-gbsn00lp, -ukai/-gkai00mp/-bkai00mp | Arphic Public License |
| wqyzenhei, wqymicrohei | fonts-wqy-zenhei, fonts-wqy-microhei | GPLv2 + font exception / Apache 2.0 |
| nanumbarungothic, nanumsquare, nanumsquareround | fonts-nanum | OFL 1.1 |
| baekmukbatang, baekmukdotum | fonts-baekmuk | Baekmuk licence |
| unbatang, undotum, ungraphic, ungungseo | fonts-unfonts-core | GPLv2 |

The fonts are only rendered to train a classifier. If you redistribute the model,
check the GPL-licensed ones (WenQuanYi Zen Hei, Un fonts) for your use.
