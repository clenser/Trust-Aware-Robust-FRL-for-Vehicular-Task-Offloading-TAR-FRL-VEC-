"""Ship the evidence behind the paper in a repository-friendly form.

The active benchmark dump is ~270 MB of per-round arrays, too large to commit.
This exports it (and the supplementary / extended runs) gzipped under
``evidence/`` together with a ``SHA256SUMS`` file that records the checksum of
the ORIGINAL json, so a reader can decompress and verify it byte for byte.

    python -m experiments.export_evidence
"""

import gzip
import hashlib
import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = os.path.join(ROOT, "evidence")
    os.makedirs(out, exist_ok=True)
    manifest = json.load(open(os.path.join(ROOT, "results", "manifest.json")))
    sums = []
    for e in manifest:
        if not (e.get("published_as_active") or e.get("kind") in ("supplementary", "extended")):
            continue
        src = os.path.join(ROOT, e["directory"], "benchmark_results.json")
        if not os.path.exists(src):
            continue
        dst = os.path.join(out, e["run_id"] + ".benchmark_results.json.gz")
        with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo)
        sums.append("%s  %s" % (_sha(src), os.path.basename(dst)[:-3]))
        print("%s -> %.1f MB" % (e["run_id"], os.path.getsize(dst) / 1e6))
    with open(os.path.join(out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
