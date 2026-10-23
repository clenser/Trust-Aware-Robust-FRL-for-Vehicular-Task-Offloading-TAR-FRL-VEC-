"""Download the complete official NGSIM export, with retries.

The USDOT export endpoint is known to be unreliable: a download can be
truncated mid-stream and have a Socrata JSON error blob appended
(``{"error": true, "message": "Internal error", "status": 500}``).  The
complete file is therefore only installed when its SHA-256 matches the
reference hash that was established by the independent verification
(byte-for-byte comparison against a clean official download).

The local copy is *never* replaced by an incomplete download.
"""
import hashlib
import os
import shutil
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST = os.path.join(ROOT, "data", "source", "us101_raw.csv")
STAGING = os.path.join(os.environ.get("TEMP", "."), "ngsim_full_download.csv")
URL = "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"
EXPECTED = "6b1516b78d60d9a4e7b90905cf4893ddadfdeb01a69e7cd6d3079e170957ecc6"
EXPECTED_SIZE = 1532183381
MAX_ATTEMPTS = 8


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def attempt(n):
    print("\n=== attempt %d/%d ===" % (n, MAX_ATTEMPTS), flush=True)
    t0 = time.time()
    req = urllib.request.Request(URL, headers={"User-Agent": "VANNET-research/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(STAGING, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        last = time.time()
        while True:
            try:
                blk = r.read(1 << 20)
            except Exception as exc:
                print("  stream error: %s" % exc)
                break
            if not blk:
                break
            out.write(blk)
            got += len(blk)
            if time.time() - last > 15:
                last = time.time()
                print("  %.1f MB / %.1f MB" % (got / 1e6, total / 1e6), flush=True)
    size = os.path.getsize(STAGING)
    print("  downloaded %.1f MB in %.0f s" % (size / 1e6, time.time() - t0))
    if size != EXPECTED_SIZE:
        print("  incomplete: %d bytes (expected %d)" % (size, EXPECTED_SIZE))
        return False
    digest = sha256(STAGING)
    print("  sha256: %s" % digest)
    if digest != EXPECTED:
        print("  hash mismatch")
        return False
    return True


def main():
    for n in range(1, MAX_ATTEMPTS + 1):
        try:
            if attempt(n):
                backup = DEST + ".truncated-backup"
                if os.path.exists(DEST):
                    shutil.move(DEST, backup)
                    print("Previous (truncated) copy kept at: %s" % backup)
                shutil.move(STAGING, DEST)
                print("Installed the complete, hash-verified export at: %s" % DEST)
                return 0
        except Exception as exc:
            print("  attempt failed: %s: %s" % (type(exc).__name__, exc))
        time.sleep(5)
    print("\nThe export endpoint did not yield a complete, hash-verified file "
          "in %d attempts. The local copy was left untouched." % MAX_ATTEMPTS)
    return 2


if __name__ == "__main__":
    sys.exit(main())
