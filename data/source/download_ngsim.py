"""
Authentic FHWA NGSIM US-101 Trajectory Dataset Download & Verification Utility.

Provides verified retrieval and SHA-256 provenance verification for authentic
FHWA Next Generation Simulation (NGSIM) US-101 vehicle trajectory data.

Metadata & Provenance:
- Issuing Agency: Federal Highway Administration (FHWA), USDOT
- Source Portal: https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj
- Direct File: the Socrata CSV export of dataset 8ect-6jqj
- Expected Record Count: 11,850,526 rows (25 columns) across four road sites
  (us-101, i-80, lankershim, peachtree), sampled at 0.1 s
- Expected Format: Imperial units (Global_Time ms, Local_X ft, Local_Y ft, v_Vel ft/s, Lane_ID)
- License: Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0),
  as listed in the official USDOT Socrata catalogue for this dataset.
- Attribution: USDOT / FHWA, 'Next Generation Simulation (NGSIM) Vehicle
  Trajectories and Supporting Data', data.transportation.gov dataset 8ect-6jqj

Authentication model
--------------------
The raw export is **not** authenticated by a hash that this repository
generates from its own copy.  The official USDOT catalogue publishes no
checksum for dataset ``8ect-6jqj`` (this was checked against the portal
metadata JSON on 2026-10-21), so a recorded hash can only ever be a
*reproducibility fingerprint*: it tells you that a later download matches the
copy you already have, but it cannot by itself prove that the copy came from
the authority.

Independent authentication therefore requires **documented verification
evidence** - a byte-for-byte comparison against a fresh download from the
official endpoint, recorded in ``verification_record.json`` (retrieval date,
source URL, portal metadata, official licence, observed sizes and hashes).

* ``--record-hash`` records a reproducibility fingerprint only.  It prints
  that the file is *not* independently authenticated.
* ``--verify`` reports the authentication status, distinguishing
  "fingerprint matches" from "authenticated against the official source".
* The official licence is *Creative Commons Attribution | Share Alike 3.0
  Unported* (CC BY-SA 3.0), as listed in the catalogue - not CC0.
"""

import os
import sys
import json
import hashlib
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = HERE
DEST_FILE = os.path.join(TARGET_DIR, "us101_raw.csv")
REFERENCE_HASH_FILE = os.path.join(TARGET_DIR, "us101_raw.sha256")
VERIFICATION_RECORD = os.path.join(TARGET_DIR, "verification_record.json")

# USDOT Socrata Open Data Direct Download Endpoint for NGSIM trajectories
USDOT_NGSIM_URL = "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"
USDOT_METADATA_URL = "https://data.transportation.gov/api/views/8ect-6jqj.json"


def compute_sha256(filepath):
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_reference_hash():
    """Returns the recorded reference SHA-256, or an empty string when absent."""
    if not os.path.exists(REFERENCE_HASH_FILE):
        return ""
    with open(REFERENCE_HASH_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line.split()[0]
    return ""


def write_reference_hash(digest):
    """Record a *reproducibility fingerprint* for the local copy.

    This is deliberately NOT called an authentication: a hash generated from
    the repository's own file can only prove that a later download matches
    that file, never that the file came from the authority.  Independent
    authentication requires the byte-for-byte comparison recorded in
    ``verification_record.json``.
    """
    with open(REFERENCE_HASH_FILE, "w") as f:
        f.write("# REPRODUCIBILITY FINGERPRINT of the local copy of the FHWA NGSIM raw export.\n")
        f.write("# This is NOT an authenticity proof: the official catalogue publishes no\n")
        f.write("# checksum for dataset 8ect-6jqj.  Independent authentication is recorded\n")
        f.write("# in data/source/verification_record.json (byte-for-byte comparison against\n")
        f.write("# a fresh download from the official endpoint).\n")
        f.write("%s  us101_raw.csv\n" % digest)


def _independent_status():
    """The recorded independent-authentication status, if any."""
    if not os.path.exists(VERIFICATION_RECORD):
        return None
    try:
        with open(VERIFICATION_RECORD, "r") as f:
            rec = json.load(f)
    except Exception:
        return None
    return rec.get("independent_authentication_status")


def _report_status(ok, reason, digest=None):
    """Report the authentication status.

    ``ok`` refers to the *reproducibility fingerprint* only.  The independent
    authentication status (established by a byte-for-byte comparison against a
    fresh official download) is reported separately so that the two are never
    conflated.
    """
    print("[AUTH-CHECK] REPRODUCIBILITY FINGERPRINT: %s"
          % ("matches the recorded reference" if ok else "does NOT match"))
    if digest:
        print("             SHA-256: %s" % digest)
    print("             Reason : %s" % reason)
    independent = _independent_status()
    if independent:
        print("             INDEPENDENT AUTHENTICATION: %s" % independent)
        print("             (see data/source/verification_record.json)")
    else:
        print("             INDEPENDENT AUTHENTICATION: unresolved - no verification "
              "record is present.")
    return ok


def verify_or_download_ngsim(force_download=False, allow_unauthenticated=False):
    """Ensures an authenticated copy of the raw NGSIM export is present.

    Returns ``(success, sha256_or_None)``.  ``success`` is True only when the
    file exists and its checksum matches the recorded reference.  A file with
    no reference hash is reported as unauthenticated and, unless
    ``allow_unauthenticated`` is set, is *not* accepted.
    """
    reference = read_reference_hash()

    if os.path.exists(DEST_FILE) and not force_download:
        actual_hash = compute_sha256(DEST_FILE)
        size_mb = os.path.getsize(DEST_FILE) / (1024 * 1024)
        print("[AUTH-CHECK] Raw NGSIM file found at: %s" % DEST_FILE)
        print("             Size   : %.2f MB" % size_mb)
        print("             SHA-256: %s" % actual_hash)

        if not reference:
            reason = ("No reference SHA-256 recorded, so the file cannot be "
                      "authenticated by checksum. Verify against the official "
                      "USDOT source, then run: "
                      "python data/source/download_ngsim.py --record-hash")
            if allow_unauthenticated:
                _report_status(False, reason + " (accepted because allow_unauthenticated=True)", actual_hash)
                return True, actual_hash
            _report_status(False, reason, actual_hash)
            return False, actual_hash

        if actual_hash == reference:
            _report_status(True, "SHA-256 matches the recorded reference hash.", actual_hash)
            return True, actual_hash

        reason = ("SHA-256 mismatch. Expected %s, observed %s. The file may be "
                  "corrupted or from a different dataset version."
                  % (reference, actual_hash))
        _report_status(False, reason, actual_hash)
        return False, actual_hash

    print("[AUTH-CHECK] Raw NGSIM dataset not present at: %s" % DEST_FILE)
    print("             Source URL: %s" % USDOT_NGSIM_URL)
    print("             Attempting automated download from the USDOT Open Data Portal...")

    try:
        def reporthook(blocknum, blocksize, totalsize):
            read = blocknum * blocksize
            if totalsize > 0:
                percent = read * 100 / totalsize
                sys.stdout.write("\rDownloading: %.1f MB / %.1f MB (%.1f%%)"
                                 % (read / (1024 * 1024), totalsize / (1024 * 1024), percent))
                sys.stdout.flush()

        urllib.request.urlretrieve(USDOT_NGSIM_URL, DEST_FILE, reporthook=reporthook)
        print("\nDownload complete. Calculating SHA-256 hash...")
        actual_hash = compute_sha256(DEST_FILE)
        print("Dataset SHA-256: %s" % actual_hash)

        if not reference:
            reason = ("Downloaded, but no reference SHA-256 is recorded so the "
                      "file cannot be authenticated. Verify it against the "
                      "official source, then run: "
                      "python data/source/download_ngsim.py --record-hash")
            if allow_unauthenticated:
                _report_status(False, reason + " (accepted because allow_unauthenticated=True)", actual_hash)
                return True, actual_hash
            _report_status(False, reason, actual_hash)
            return False, actual_hash

        if actual_hash == reference:
            _report_status(True, "SHA-256 matches the recorded reference hash.", actual_hash)
            return True, actual_hash

        _report_status(False, "SHA-256 mismatch against the recorded reference hash.", actual_hash)
        return False, actual_hash

    except Exception as e:
        print("\n[BLOCKED] Automated download could not be completed (%s)." % e)
        print("          Please download the NGSIM trajectory export from:")
        print("          %s" % USDOT_NGSIM_URL)
        print("          and save it to: %s" % DEST_FILE)
        return False, None


if __name__ == "__main__":
    if "--record-hash" in sys.argv:
        if not os.path.exists(DEST_FILE):
            print("Cannot record a hash: %s does not exist." % DEST_FILE)
            sys.exit(1)
        digest = compute_sha256(DEST_FILE)
        write_reference_hash(digest)
        print("[AUTH-CHECK] Reproducibility fingerprint recorded: %s" % digest)
        print("             This is NOT an authenticity proof: the official catalogue")
        print("             publishes no checksum for dataset 8ect-6jqj, so a hash")
        print("             generated from this repository's own copy can only show")
        print("             that a later download matches it.")
        print("             Independent authentication requires a byte-for-byte")
        print("             comparison against a fresh official download, recorded in")
        print("             data/source/verification_record.json.")
        sys.exit(0)

    allow_unauthenticated = "--allow-unauthenticated" in sys.argv
    ok, digest = verify_or_download_ngsim(allow_unauthenticated=allow_unauthenticated)
    sys.exit(0 if ok else 2)
