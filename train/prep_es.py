"""Spanish ASR data: download from Hugging Face, convert to 16 kHz mono FLAC, write jsonl manifests.

Sources (all CC0 or CC-BY, which allow a released model under CC BY-NC-SA 4.0):
  Common Voice 17 es (CC0, HF mirror fsicoli/common_voice_17_0), VoxPopuli es (CC0), Multilingual LibriSpeech es
  (CC-BY-4.0), FLEURS es_419 (CC-BY-4.0).

python prep_es.py download --root /opt/dlami/nvme/es
python prep_es.py convert  --root /opt/dlami/nvme/es --procs 8
python prep_es.py download_other / convert_other   # Common Voice's unvalidated clips (+1,300 h; filter with filter_teacher.py)
Manifests: <root>/manifests/{train,dev,test}_<src>.jsonl with {"wav","text","duration","source"}.
"""
import argparse, csv, glob, io, json, os, sys, tarfile
from multiprocessing import Pool

import numpy as np
import soundfile as sf

csv.field_size_limit(sys.maxsize)


def to16k(a, sr):
    if a.ndim > 1:
        a = a.mean(1)
    if sr != 16000:
        import soxr
        a = soxr.resample(a, sr, 16000)
    return np.clip(a, -1.0, 1.0)


def write_flac(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, a, 16000, format="FLAC", subtype="PCM_16")
    return len(a) / 16000.0


def download(root):
    from huggingface_hub import snapshot_download
    raw = os.path.join(root, "raw")
    jobs = [
        ("fsicoli/common_voice_17_0", ["audio/es/train/*", "audio/es/dev/*", "audio/es/test/*",
                                       "transcript/es/train.tsv", "transcript/es/dev.tsv", "transcript/es/test.tsv"]),
        ("facebook/voxpopuli", ["es/*.parquet"]),
        ("facebook/multilingual_librispeech", ["spanish/train-*", "spanish/dev-*", "spanish/test-*"]),
        ("google/fleurs", ["data/es_419/audio/*", "data/es_419/*.tsv"]),
    ]
    for repo, pats in jobs:
        print("downloading", repo, flush=True)
        snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=raw, max_workers=16)


# ---------------------------------------------------------------- converters (one task = one shard)
def conv_cv_tar(args):
    tar_path, split, sents, out = args
    rows = []
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if name not in sents:
                continue
            try:
                a, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                p = os.path.join(out, "cv", split, name.replace(".mp3", ".flac"))
                d = write_flac(p, to16k(a, sr))
                rows.append({"wav": p, "text": sents[name], "duration": round(d, 3), "source": "cv"})
            except Exception as e:
                print("cv bad", name, e, flush=True)
    return rows


def conv_parquet(args):
    pq_path, src, split, out = args
    import pyarrow.parquet as pq
    rows = []
    for batch in pq.ParquetFile(pq_path).iter_batches(batch_size=128):  # stream: VoxPopuli shards are 3 GB
      for r in batch.to_pylist():
        try:
            if src == "vox":
                key, text = r["audio_id"], r["normalized_text"]
            else:  # mls
                key, text = r["id"], r["transcript"]
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            p = os.path.join(out, src, split, key.replace(":", "_") + ".flac")
            d = write_flac(p, to16k(a, sr))
            rows.append({"wav": p, "text": text, "duration": round(d, 3), "source": src})
        except Exception as e:
            print(src, "bad", e, flush=True)
    return rows


def conv_fleurs(args):
    tgz, split, sents, out = args
    rows = []
    with tarfile.open(tgz) as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if name not in sents:
                continue
            a, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            p = os.path.join(out, "fleurs", split, name.replace(".wav", ".flac"))
            d = write_flac(p, to16k(a, sr))
            rows.append({"wav": p, "text": sents[name], "duration": round(d, 3), "source": "fleurs"})
    return rows


_SENTS = {}  # filled before the pool forks (the "other" transcript table is too big to pickle per task)


def conv_cv_tar_g(args):
    tar_path, split, out = args
    return conv_cv_tar((tar_path, split, _SENTS, out))


def download_other(root):
    from huggingface_hub import snapshot_download
    snapshot_download("fsicoli/common_voice_17_0", repo_type="dataset", local_dir=os.path.join(root, "raw"), max_workers=16,
                      allow_patterns=["audio/es/other/*", "transcript/es/other.tsv"])


def convert_other(root, procs):
    raw, out = os.path.join(root, "raw"), os.path.join(root, "audio")
    with open(f"{raw}/transcript/es/other.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            _SENTS[r["path"]] = r["sentence"]
    print(len(_SENTS), "transcripts", flush=True)
    tars = sorted(glob.glob(f"{raw}/audio/es/other/*.tar"))
    os.makedirs(os.path.join(root, "manifests"), exist_ok=True)
    with Pool(procs) as pool, open(os.path.join(root, "manifests", "train_cvother.jsonl"), "w") as f:
        for i, rows in enumerate(pool.imap_unordered(conv_cv_tar_g, [(t, "other", out) for t in tars])):
            for row in rows:
                row["source"] = "cvother"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(tars)}] +{len(rows)} utts, {sum(x['duration'] for x in rows)/3600:.1f} h", flush=True)


def convert(root, procs):
    raw, out = os.path.join(root, "raw"), os.path.join(root, "audio")
    man = os.path.join(root, "manifests")
    os.makedirs(man, exist_ok=True)
    tasks = []  # (fn, args, manifest name)
    for split in ["train", "dev", "test"]:
        sents = {}
        with open(f"{raw}/transcript/es/{split}.tsv", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                sents[r["path"]] = r["sentence"]
        for tp in sorted(glob.glob(f"{raw}/audio/es/{split}/*.tar")):
            tasks.append((conv_cv_tar, (tp, split, sents, out), f"{split}_cv"))
    for f in sorted(glob.glob(f"{raw}/es/*.parquet")):
        split = os.path.basename(f).split("-")[0].replace("validation", "dev")
        tasks.append((conv_parquet, (f, "vox", split, out), f"{split}_vox"))
    for f in sorted(glob.glob(f"{raw}/spanish/*.parquet")):
        split = os.path.basename(f).split("-")[0]
        if split in ("train", "dev", "test"):
            tasks.append((conv_parquet, (f, "mls", split, out), f"{split}_mls"))
    for split in ["train", "dev", "test"]:
        sents = {}
        with open(f"{raw}/data/es_419/{split}.tsv", newline="") as f:
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                sents[r[1]] = r[3]  # id, file_name, raw_transcription, transcription, ...
        tasks.append((conv_fleurs, (f"{raw}/data/es_419/audio/{split}.tar.gz", split, sents, out), f"{split}_fleurs"))
    print(len(tasks), "shards", flush=True)
    files = {}
    with Pool(procs) as pool:
        res = [(name, pool.apply_async(fn, (a,))) for fn, a, name in tasks]
        for i, (name, r) in enumerate(res):
            rows = r.get()
            if name not in files:
                files[name] = open(os.path.join(man, name + ".jsonl"), "w")
            for row in rows:
                files[name].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[name].flush()
            print(f"[{i+1}/{len(res)}] {name}: +{len(rows)} utts, {sum(x['duration'] for x in rows)/3600:.1f} h", flush=True)
    for f in files.values():
        f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "convert", "download_other", "convert_other"])
    ap.add_argument("--root", default="/opt/dlami/nvme/es")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    {"download": lambda: download(a.root), "convert": lambda: convert(a.root, a.procs),
     "download_other": lambda: download_other(a.root), "convert_other": lambda: convert_other(a.root, a.procs)}[a.cmd]()
