"""WER of the deployed C engine (host build == ESP32 arithmetic) on LibriSpeech and other test sets.

python eval_engine.py model.tasr [--sets test-clean test-other] [--chunk 32 --left 4] [--limit N] [--manifest x.jsonl]
python eval_engine.py es.tnm --sets --lang es --manifest test_cv.jsonl ...      # Spanish normalization (data.py)
python eval_engine.py es.tnm --stream [--chunk 16 --left 128] ...                 # streaming engine for .tnm models
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np, soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "train"))
from wer_utils import load_librispeech, wer  # noqa: E402

_stream = None


def _init(path, chunk, left, lm_path, beam, lw, tb, stream=False, sc=0, sl=0, topk=6):
    global _stream
    import pytasr
    lm = pytasr.LM(lm_path) if lm_path else None
    if path.endswith(".tnm"):  # NeMo conformer-ctc-small utterance engine (or its streaming engine)
        _stream = pytasr.Nemo(path, lm=lm, beam=beam, topk=topk, lm_weight=lw, token_bonus=tb)
        if stream:
            _stream = pytasr.NemoStream(_stream, sc, sl)  # 0 = the chunk / left context stored in the model
    else:
        _stream = pytasr.Stream(pytasr.Model(path), chunk, left, lm=lm, beam=beam, lm_weight=0.5 if lw is None else lw, token_bonus=1.0 if tb is None else tb)


def _run(item):
    uid, path, ref = item
    a, sr = sf.read(path, dtype="int16")
    if a.ndim > 1:
        a = a[:, 0]
    import pytasr
    if isinstance(_stream, pytasr.Nemo):
        return uid, _stream.transcribe(a), ref
    if isinstance(_stream, pytasr.NemoStream):  # transcript does not depend on the feed size (verified)
        _stream.reset()
        _stream.feed(a)
        return uid, _stream.finish(), ref
    return uid, _stream.transcribe(a, feed=320), ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sets", nargs="*", default=["test-clean", "test-other"])
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--left", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--dump", default="")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--stream", action="store_true", help="streaming engine (models exported with --stream)")
    ap.add_argument("--sc", type=int, default=0, help="--stream: chunk in encoder frames (0 = model default)")
    ap.add_argument("--sl", type=int, default=0, help="--stream: left context in encoder frames (0 = model default)")
    ap.add_argument("--lm", default="")
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--topk", type=int, default=6, help="candidate tokens per frame in the beam search")
    ap.add_argument("--lm_weight", type=float, default=None, help="default: the weights stored in the LM file")
    ap.add_argument("--token_bonus", type=float, default=None)
    a = ap.parse_args()
    def sub(items):
        return items[:: max(1, len(items) // a.limit)][: a.limit] if a.limit else items
    jobs = [(s, sub(load_librispeech(s))) for s in a.sets]
    score = wer
    if a.lang != "en":  # references and hypotheses with the language's training normalization
        import jiwer
        from data import CLEANERS
        clean = CLEANERS[a.lang]

        def score(refs, hyps):
            o = jiwer.process_words(refs, [clean(h) or "-" for h in hyps])
            return 100 * o.wer, o
    for m in a.manifest:
        its = [json.loads(l) for l in open(m)]
        if a.lang != "en":
            its = [dict(d, text=clean(d["text"])) for d in its]
            its = [d for d in its if d["text"]]
        if a.limit:
            its = its[: a.limit]
        jobs.append((os.path.basename(m), [(d.get("id", str(i)), d["wav"], d["text"]) for i, d in enumerate(its)]))
    res = {}
    with Pool(a.procs, initializer=_init,
              initargs=(a.model, a.chunk, a.left, a.lm, a.beam, a.lm_weight, a.token_bonus, a.stream, a.sc, a.sl, a.topk)) as pool:
        for name, items in jobs:
            t = time.time()
            out = pool.map(_run, items, chunksize=4)
            w, o = score([r for _, _, r in out], [h for _, h, _ in out])
            dur = sum(sf.info(p).duration for _, p, _ in items)
            print(f"{name}: WER {w:.2f}% (sub {o.substitutions} del {o.deletions} ins {o.insertions}) "
                  f"{len(items)} utts {dur/3600:.2f}h in {time.time()-t:.0f}s", flush=True)
            res[name] = w
            if a.dump:
                with open(f"{a.dump}.{name}.txt", "w") as f:
                    for uid, h, r in out:
                        f.write(f"{uid}\tREF: {r}\n{uid}\tHYP: {h}\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
