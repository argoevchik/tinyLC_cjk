"""Train TinyFC-CJK on synthetic renders.

  ./run.sh -m tinyfc_cjk.train --cv              # design-group-held-out cross-validation
  ./run.sh -m tinyfc_cjk.train --steps 15000     # final model on all fonts -> models/tinyfc_cjk.pt
"""

import argparse
import json
import random
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .model import TRAIT_MEAN, TRAIT_STD, TinyFC, to_tensor
from .render import charsets, discover_variants, make_sample
from .taxonomy import COARSE, FINE_HEADS, HEAD_KEYS, LANGS, ROOT, TRAIT_CUTS, cv_folds, trait_class


def sampling_tree(variants):
    """coarse -> lang -> fine class ('' if none) -> group -> [variants]."""
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for v in variants:
        for lang in v.font.langs:
            fc = v.font.fine.get(lang, "") if (lang, v.font.coarse) in FINE_HEADS else ""
            tree[v.font.coarse][lang][fc][v.font.group].append(v)
    return tree


class SynthStream(IterableDataset):
    """Balanced: coarse class, then language, then fine class, then design group,
    then font face. Rendering happens on the fly in the workers."""

    def __init__(self, variants, cs, seed=0):
        self.tree, self.cs, self.seed = sampling_tree(variants), cs, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0) + random.getrandbits(20))
        pick = lambda d: d[rng.choice(sorted(d))]
        while True:
            c = rng.choice(sorted(self.tree))
            lang = rng.choice(sorted(self.tree[c]))
            v = rng.choice(pick(pick(self.tree[c][lang])))
            img, *labels = make_sample(v, lang, self.cs[v.key], rng)
            yield img, *labels


def _eval_job(args):
    variants, lang, n, seed, cs = args
    rng = random.Random(seed)
    out = []
    for i in range(n):
        v = variants[i % len(variants)]
        img, *lab = make_sample(v, lang, cs[v.key], rng)
        out.append((img, *lab))
    return out


def make_eval_set(variants, cs, per_unit=120, seed=1234, procs=16):
    """per_unit crops for every (font, language) of the given variants."""
    by_unit = defaultdict(list)
    for v in variants:
        for lang in v.font.langs:
            by_unit[(v.font.id, lang)].append(v)
    units = sorted(by_unit)
    jobs = [(by_unit[u], u[1], per_unit, seed + i, {v.key: cs[v.key] for v in by_unit[u]}) for i, u in enumerate(units)]
    with Pool(procs) as p:
        res = p.map(_eval_job, jobs)
    xs, labels, meta = [], [], []
    fonts = {v.font.id: v.font for v in variants}
    for u, rows in zip(units, res):
        for img, *lab in rows:
            xs.append(img)
            labels.append(lab)
            meta.append((u[0], fonts[u[0]].group, u[1]))
    return np.stack(xs), np.array(labels), meta


@torch.no_grad()
def predict_batches(model, xs, device, bs=512):
    model.eval()
    pc, ps, pf, pt = [], [], [[] for _ in HEAD_KEYS], []
    for i in range(0, len(xs), bs):
        c, s, f, t = model(to_tensor(xs[i: i + bs]).to(device))
        pt.append((t.float() * TRAIT_STD + TRAIT_MEAN).cpu())
        pc.append(F.softmax(c.float(), 1).cpu())
        ps.append(F.softmax(s.float(), 1).cpu())
        for k, fk in enumerate(f):
            pf[k].append(F.softmax(fk.float(), 1).cpu())
    return torch.cat(pc).numpy(), torch.cat(ps).numpy(), [torch.cat(p).numpy() for p in pf], torch.cat(pt).numpy()


def evaluate(model, xs, labels, meta, device):
    pc, ps, pf, pt = predict_batches(model, xs, device)
    coarse, script, head, fcls = labels[:, :4].astype(int).T
    ratio = labels[:, 4]
    font_ids = np.array([m[0] for m in meta])
    langs = np.array([m[2] for m in meta])
    res = dict(n=len(xs), coarse_crop_acc=float((pc.argmax(1) == coarse).mean()),
               script_acc=float((ps.argmax(1) == script).mean()), per_lang={}, fonts={}, fine={})
    for lang in LANGS:
        m = langs == lang
        if m.any():
            res["per_lang"][lang] = dict(coarse_crop_acc=round(float((pc[m].argmax(1) == coarse[m]).mean()), 3), n=int(m.sum()))
    # font level: average the probabilities of all crops of one (font, language)
    for fid, lang in sorted(set(zip(font_ids, langs))):
        m = (font_ids == fid) & (langs == lang)
        true = COARSE[coarse[m][0]]
        mp = pc[m].mean(0)
        r = dict(lang=lang, group=meta[int(np.argmax(m))][1], true=true, pred=COARSE[int(mp.argmax())],
                 p=round(float(mp.max()), 2), crop_acc=round(float((pc[m].argmax(1) == coarse[m]).mean()), 3))
        hm = m & (head >= 0)
        if hm.any():
            k = int(head[hm][0])
            classes = FINE_HEADS[HEAD_KEYS[k]]
            fp = pf[k][hm]
            r["fine_true"] = classes[int(fcls[hm][0])]
            r["fine_pred"] = classes[int(fp.mean(0).argmax())]
            r["fine_crop_acc"] = round(float((fp.argmax(1) == fcls[hm]).mean()), 3)
        # no language given: route by the predicted script and predicted coarse class, all crops
        r["script_pred"] = LANGS[int(ps[m].mean(0).argmax())]
        akey = (r["script_pred"], r["pred"])
        if akey in FINE_HEADS:
            k = HEAD_KEYS.index(akey)
            r["auto_fine"] = FINE_HEADS[akey][int(pf[k][m].mean(0).argmax())]
        tm = m & ~np.isnan(ratio)
        if tm.any():  # kana-ratio regression: mean over crops, then the taxonomy's cuts
            r["ratio_true"] = round(float(ratio[tm][0]), 3)
            r["ratio_pred"] = round(float(np.median(pt[tm])), 3)
            if (lang, true) in TRAIT_CUTS:
                r["trait_pred"] = trait_class((lang, true), r["ratio_pred"])
        res["fonts"][f"{fid}/{lang}"] = r
    fonts = list(res["fonts"].values())
    res["coarse_font_acc"] = float(np.mean([f["true"] == f["pred"] for f in fonts]))
    for k, key in enumerate(HEAD_KEYS):
        m = head == k
        fk = [f for f in fonts if f.get("fine_true") and (f["lang"], f["true"]) == key]
        if m.any():
            res["fine"]["/".join(key)] = dict(
                crop_acc=round(float((pf[k][m].argmax(1) == fcls[m]).mean()), 3),
                font_correct=f"{sum(f['fine_true'] == f['fine_pred'] for f in fk)}/{len(fk)}")
            if key in TRAIT_CUTS:
                res["fine"]["/".join(key)]["trait_font_correct"] = f"{sum(f['fine_true'] == f.get('trait_pred') for f in fk)}/{len(fk)}"
    jr = [f for f in fonts if "ratio_pred" in f]
    if jr:
        a, b = np.array([f["ratio_true"] for f in jr]), np.array([f["ratio_pred"] for f in jr])
        res["kana_ratio"] = dict(mae=round(float(np.abs(a - b).mean()), 4),
                                 corr=round(float(np.corrcoef(a, b)[0, 1]), 3) if len(jr) > 2 else None)
    conf = np.zeros((len(COARSE), len(COARSE)), int)
    for t, p in zip(coarse, pc.argmax(1)):
        conf[t, p] += 1
    res["coarse_confusion"] = conf.tolist()
    return res


def train(variants, cs, steps, device, eval_data=None, bs=128, lr=3e-3, workers=16, log=print, trait_w=1.0):
    model = TinyFC().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    dl = DataLoader(SynthStream(variants, cs), batch_size=bs, num_workers=workers,
                    persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    t0, run = time.time(), defaultdict(float)
    for step in range(1, steps + 1):
        model.train()
        x, yc, ys, yh, yf, yt = next(it)
        x = to_tensor(x.numpy()).to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        yc, ys, yh, yf = (t.to(device) for t in (yc, ys, yh, yf))
        yt = ((yt.float() - TRAIT_MEAN) / TRAIT_STD).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            oc, os_, of, ot = model(x)
            lc = F.cross_entropy(oc, yc, label_smoothing=0.1)
            ls = F.cross_entropy(os_, ys, label_smoothing=0.1)
            lf = torch.zeros((), device=device)
            for k, o in enumerate(of):
                m = yh == k
                if m.any():
                    lf = lf + F.cross_entropy(o[m], yf[m], label_smoothing=0.1) * m.float().mean() * len(of)
            tm = ~torch.isnan(yt)
            lt = F.smooth_l1_loss(ot.float()[tm], yt[tm]) * tm.float().mean() * 2 if tm.any() else torch.zeros((), device=device)
            loss = lc + 0.3 * ls + lf + trait_w * lt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run["loss"] += loss.item(); run["coarse"] += lc.item(); run["fine"] += lf.item(); run["trait"] += lt.item()
        if step % 500 == 0 or step == steps:
            msg = f"step {step}/{steps} " + " ".join(f"{k} {v / 500:.3f}" for k, v in run.items()) + f" {time.time() - t0:.0f}s"
            run.clear()
            if eval_data is not None and (step % 2000 == 0 or step == steps):
                r = evaluate(model, *eval_data, device)
                msg += f" | held-out coarse crop {r['coarse_crop_acc']:.3f} font {r['coarse_font_acc']:.2f} " + \
                       " ".join(f"{k} {v['crop_acc']:.2f}" for k, v in r["fine"].items()) + f" kana_ratio {r.get('kana_ratio')}"
            log(msg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", default=str(ROOT / "models" / "tinyfc_cjk.pt"))
    ap.add_argument("--trait-w", type=float, default=1.0, help="weight of the kana-ratio loss (0 = off)")
    ap.add_argument("--tag", default="", help="suffix for results/cv<tag>.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda m: print(m, flush=True)
    variants = discover_variants()
    cs = charsets()
    log(f"{len(variants)} faces from {len({v.font.id for v in variants})} fonts, "
        f"{len({v.font.group for v in variants})} design groups; fine heads: {HEAD_KEYS}")
    if a.cv:
        out = ROOT / "results" / f"cv{a.tag}.json"
        out.parent.mkdir(exist_ok=True)
        results = json.loads(out.read_text()) if out.exists() and a.only_fold is not None else {}
        for k, held in enumerate(cv_folds(a.folds)):
            if a.only_fold is not None and k != a.only_fold:
                continue
            log(f"\n=== fold {k}: held-out groups {held}")
            tr = [v for v in variants if v.font.group not in held]
            te = [v for v in variants if v.font.group in held]
            ev = make_eval_set(te, cs)
            model = train(tr, cs, a.steps, device, ev, workers=a.workers, log=log, trait_w=a.trait_w)
            r = evaluate(model, *ev, device)
            r["held_out"] = held
            torch.save({"state_dict": model.state_dict(), "coarse": COARSE, "langs": LANGS, "held_out": held,
                        "fine_heads": {"/".join(k): v for k, v in FINE_HEADS.items()}},
                       ROOT / "models" / f"cv_fold{k}{a.tag}.pt")
            for name, fr in r["fonts"].items():
                ok = "OK" if fr["true"] == fr["pred"] else "XX"
                fine = ""
                if "fine_true" in fr:
                    fine = f"  fine {'OK' if fr['fine_true'] == fr['fine_pred'] else 'XX'} {fr['fine_true']} -> {fr['fine_pred']} ({fr['fine_crop_acc']})"
                if "ratio_pred" in fr:
                    fine += f"  kana_ratio {fr['ratio_true']} -> {fr['ratio_pred']}" + (f" => {fr['trait_pred']}" if "trait_pred" in fr else "")
                log(f"  {ok} {name:28s} {fr['true']:8s} -> {fr['pred']:8s} p={fr['p']:.2f} crop={fr['crop_acc']}{fine}")
            log(f"  fold {k}: coarse crop {r['coarse_crop_acc']:.3f}, font {r['coarse_font_acc']:.3f}, script {r['script_acc']:.3f}, fine {r['fine']}")
            results[str(k)] = r
            out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        log(f"saved {out}")
    else:
        model = train(variants, cs, a.steps, device, workers=a.workers, log=log, trait_w=a.trait_w)
        Path(a.out).parent.mkdir(exist_ok=True, parents=True)
        torch.save({"state_dict": model.state_dict(), "coarse": COARSE, "langs": LANGS,
                    "fine_heads": {"/".join(k): v for k, v in FINE_HEADS.items()}}, a.out)
        log(f"saved {a.out} {sum(p.numel() for p in model.parameters())} params")


if __name__ == "__main__":
    main()
