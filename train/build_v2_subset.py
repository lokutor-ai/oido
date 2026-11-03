"""Sample the ~1,000 h fine-tuning mix for the Oido v2 variants from the en_stream2 training manifest (train_v2b),
dropping every utterance whose text equals a held-out text (LibriSpeech dev/test, data/val/*.jsonl).

python build_v2_subset.py --base train_v2b.jsonl.gz --out train_v2_sub.jsonl.gz [--val_dir ../data/val --ls_dir ../data/LibriSpeech]
"""
import argparse, collections, glob, gzip, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from wer_utils import normalize  # noqa: E402

CAPS_H = {"librispeech-v2.1-en": 80, "librispeech-v2.1-en_360": 140, "librispeech-other-500": 140, "commonvoice17-en": 250,
          "ps-clean-teacher": 200, "ps-dirty-teacher": 80, "mls-en-big": 60, "ami-ihm": 50, "vctk": 20, "voxpopuli-en": 30,
          "openslr70-en-ng": 4, "openslr83-en-uk-ie": 24}


# ~1,900 h mix (superset of the 1,077 h one for the same seed: same shuffled order per source, longer prefix)
CAPS_H_FULL = {"librispeech-v2.1-en": 80, "librispeech-v2.1-en_360": 282, "librispeech-other-500": 497, "commonvoice17-en": 400,
               "ps-clean-teacher": 300, "ps-dirty-teacher": 120, "mls-en-big": 100, "ami-ihm": 60, "vctk": 40, "voxpopuli-en": 30,
               "openslr70-en-ng": 4, "openslr83-en-uk-ie": 24}


def held_texts(val_dir, ls_dir):
    held = set()
    for p in glob.glob(os.path.join(val_dir, "*.jsonl")):
        for l in open(p):
            held.add(normalize(json.loads(l)["text"]))
    for split in ["dev-clean", "test-clean", "test-other"]:
        for t in glob.glob(os.path.join(ls_dir, split, "*", "*", "*.trans.txt")):
            for l in open(t):
                held.add(normalize(l.strip().split(" ", 1)[1]))
    return held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_dir", default=os.path.join(HERE, "..", "data", "val"))
    ap.add_argument("--ls_dir", default=os.path.join(HERE, "..", "data", "LibriSpeech"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", default="sub", choices=["sub", "full"], help="sub: ~1,077 h; full: ~1,900 h")
    a = ap.parse_args()
    held = held_texts(a.val_dir, a.ls_dir)
    print("held-out texts", len(held))
    by_src, dropped = collections.defaultdict(list), collections.Counter()
    for l in gzip.open(a.base, "rt"):
        d = json.loads(l)
        if not (0.3 <= d["duration"] <= 20.0):
            continue
        if normalize(d["text"]) in held:
            dropped[d["source"]] += 1
            continue
        by_src[d["source"]].append(d)
    rng = random.Random(a.seed)
    out, hours = [], collections.Counter()
    for src, items in by_src.items():
        rng.shuffle(items)
        cap = (CAPS_H_FULL if a.mix == "full" else CAPS_H).get(src, 0) * 3600
        tot = 0.0
        for d in items:
            if tot >= cap:
                break
            out.append(d)
            tot += d["duration"]
        hours[src] = tot / 3600
    rng.shuffle(out)
    with gzip.open(a.out, "wt") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    print("dropped as held-out:", dict(dropped))
    print({k: round(v, 1) for k, v in hours.items()}, "total h", round(sum(hours.values()), 1), "utts", len(out))


if __name__ == "__main__":
    main()
