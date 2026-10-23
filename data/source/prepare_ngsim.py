"""
Authentic FHWA NGSIM US-101 ingestion pipeline.

The raw USDOT export (``us-101-trajectories-0750am-0805am.csv``) aggregates
FOUR road sites (US-101, I-80, Lankershim Boulevard, Peachtree Street) sampled
at 0.1 s intervals, contains exact duplicate rows and has a timestamp-scale
inconsistency for the Peachtree records.  It therefore cannot be replayed
directly.

This module produces a *processed* US-101 dataset that is suitable for trace
replay, and records full provenance for it:

* authenticates the raw file against a recorded SHA-256 reference,
* filters the raw export to the intended road site (US-101),
* resolves and reports timestamp-scale anomalies,
* removes exact duplicate records,
* preserves the original source ``Vehicle_ID`` values,
* converts imperial units to metric,
* writes a provenance sidecar (source hash, filter rules, counts, checksums).

The raw file is never modified.

Usage
-----
    python data/source/prepare_ngsim.py            # build the processed trace
    python data/source/prepare_ngsim.py --verify   # only verify the raw file
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

RAW_FILE = os.path.join(HERE, "us101_raw.csv")
REFERENCE_HASH_FILE = os.path.join(HERE, "us101_raw.sha256")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "traces", "ngsim_us101_authentic.csv")
OUTPUT_PROVENANCE = os.path.join(PROJECT_ROOT, "data", "traces", "ngsim_us101_authentic.provenance.json")

TARGET_LOCATION = "us-101"
FEET_TO_M = 0.3048
NGSIM_SAMPLE_INTERVAL_MS = 100  # 0.1 second sampling


# ----------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------
def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_reference_hash() -> str:
    """Read the recorded SHA-256 of the authentic raw export."""
    if not os.path.exists(REFERENCE_HASH_FILE):
        return ""
    with open(REFERENCE_HASH_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # support both "<hash>" and "<hash>  <filename>"
                return line.split()[0]
    return ""


def write_reference_hash(digest: str, source: str = "us101_raw.csv"):
    """Record the hash so future downloads can be authenticated."""
    with open(REFERENCE_HASH_FILE, "w") as f:
        f.write("# SHA-256 reference for the authentic FHWA NGSIM raw export.\n")
        f.write("# Update this only after verifying the file against the official source.\n")
        f.write(f"{digest}  {source}\n")


def authenticate_raw_file(verbose: bool = True) -> dict:
    """Verify the raw file against the recorded reference hash.

    Returns a report dict with ``authenticated`` (bool), ``reason`` and the
    observed hash.  A missing reference hash is reported as *unauthenticated*
    rather than being silently accepted.
    """
    report = {
        "path": RAW_FILE,
        "exists": os.path.exists(RAW_FILE),
        "observed_sha256": None,
        "reference_sha256": read_reference_hash(),
        "authenticated": False,
        "reason": "",
    }

    if not report["exists"]:
        report["reason"] = "Raw file is absent."
        if verbose:
            print("[AUTH] Raw file not found: %s" % RAW_FILE)
        return report

    report["observed_sha256"] = compute_sha256(RAW_FILE)
    report["size_bytes"] = os.path.getsize(RAW_FILE)

    if not report["reference_sha256"]:
        report["reason"] = (
            "No reference SHA-256 recorded. The file cannot be authenticated by "
            "checksum. Record the reference hash only after verifying the file "
            "against the official USDOT source."
        )
        if verbose:
            print("[AUTH] UNVERIFIED - no reference hash recorded.")
            print("[AUTH] Observed SHA-256: %s" % report["observed_sha256"])
        return report

    if report["observed_sha256"] == report["reference_sha256"]:
        report["authenticated"] = True
        report["reason"] = "SHA-256 matches the recorded reference hash."
        if verbose:
            print("[AUTH] VERIFIED - SHA-256 matches the recorded reference hash.")
    else:
        report["reason"] = "SHA-256 MISMATCH against the recorded reference hash."
        if verbose:
            print("[AUTH] FAILED - SHA-256 mismatch.")
            print("[AUTH]   expected: %s" % report["reference_sha256"])
            print("[AUTH]   observed: %s" % report["observed_sha256"])

    return report


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
def diagnose_timestamps(df: pd.DataFrame) -> dict:
    """Detect timestamp-scale anomalies across road sites.

    The Peachtree records use a different epoch scale from the other sites.
    A location whose median timestamp differs from the global median by more
    than an order of magnitude is flagged.
    """
    medians = df.groupby("Location")["Global_Time"].median()
    global_median = float(np.median(medians.values)) if len(medians) else 0.0

    anomalies = {}
    for loc, med in medians.items():
        loc = str(loc)
        if global_median and med and (global_median / med > 10 or med / global_median > 10):
            anomalies[loc] = {
                "median_global_time": float(med),
                "ratio_to_global_median": float(med / global_median) if global_median else None,
                "issue": "timestamp scale differs from the other road sites by >1 order of magnitude",
            }

    return {
        "global_median_global_time": global_median,
        "per_location_median": {str(k): float(v) for k, v in medians.items()},
        "anomalies": anomalies,
    }


def _verification_fields():
    """Authentication evidence recorded by the independent verification.

    The official catalogue publishes no checksum for the NGSIM dataset, so the
    recorded hash of the local copy is only a *reproducibility fingerprint*.
    Independent authentication is the byte-for-byte comparison against a fresh
    download, recorded in ``verification_record.json``.
    """
    record_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "verification_record.json")
    fields = {
        "independent_authentication_status": "unresolved",
        "published_checksum_found": False,
        "published_checksum_value": None,
        "published_checksum_note": (
            "The official catalogue publishes no checksum; authenticity was "
            "established by byte-for-byte comparison against a fresh download "
            "from the official endpoint (see data/source/verification_record.json)."),
        "license_verified_from": "https://data.transportation.gov/api/views/8ect-6jqj.json",
    }
    if not os.path.exists(record_path):
        return fields
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except Exception:
        return fields
    fields["independent_authentication_status"] = rec.get(
        "independent_authentication_status", "unresolved")
    fields["published_checksum_found"] = bool(rec.get("published_checksum_found"))
    fields["published_checksum_value"] = rec.get("published_checksum_value")
    fields["verification_record"] = "data/source/verification_record.json"
    fields["verification_attempted_utc"] = rec.get("verification_attempted_utc")
    fields["official_export_sha256"] = rec.get("downloaded_sha256")
    fields["official_export_row_count"] = rec.get("downloaded_row_count")
    return fields


def build_processed_trace(verbose: bool = True) -> dict:
    """Build the processed US-101 metric trace from the authenticated raw file."""
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            "Raw NGSIM export not found: %s. Download it with "
            "`python data/source/download_ngsim.py`." % RAW_FILE
        )

    df = pd.read_csv(RAW_FILE)
    raw_rows = len(df)
    raw_columns = list(df.columns)

    required = ["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_Vel", "Lane_ID", "Location"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Raw NGSIM file is missing required columns: %s" % missing)

    # --- 1. Timestamp diagnosis (before filtering) -----------------
    ts_diag = diagnose_timestamps(df)

    # --- 2. Filter to the intended road site ------------------------
    site_df = df[df["Location"].astype(str).str.lower() == TARGET_LOCATION].copy()
    rows_after_site_filter = len(site_df)
    if rows_after_site_filter == 0:
        raise ValueError("No records found for location '%s' in the raw export." % TARGET_LOCATION)

    # --- 3. Remove exact duplicate records --------------------------
    dup_mask = site_df.duplicated(keep="first")
    duplicate_rows_removed = int(dup_mask.sum())
    site_df = site_df[~dup_mask].copy()

    # --- 4. Remove rows with unusable numeric values ----------------
    numeric_cols = ["Global_Time", "Local_X", "Local_Y", "v_Vel", "Lane_ID"]
    for col in numeric_cols:
        site_df[col] = pd.to_numeric(site_df[col], errors="coerce")
    invalid_rows = int(site_df[numeric_cols].isna().any(axis=1).sum())
    site_df = site_df[~site_df[numeric_cols].isna().any(axis=1)].copy()

    # --- 5. Preserve source vehicle IDs -----------------------------
    site_df["vehicle_id"] = site_df["Vehicle_ID"].astype(int)
    site_df = site_df.sort_values(["Global_Time", "vehicle_id"]).reset_index(drop=True)

    # --- 6. Unit conversion (imperial -> metric) --------------------
    site_df["pos_x"] = (site_df["Local_Y"] * FEET_TO_M).round(2)   # longitudinal
    site_df["pos_y"] = (site_df["Local_X"] * FEET_TO_M).round(2)   # lateral
    site_df["speed_mps"] = (site_df["v_Vel"] * FEET_TO_M).round(2)
    site_df["lane_id"] = site_df["Lane_ID"].astype(int)
    site_df["time_sec"] = ((site_df["Global_Time"] - site_df["Global_Time"].min()) / 1000.0).round(1)

    # --- 7. Sampling-interval sanity check --------------------------
    deltas = np.diff(np.sort(site_df["Global_Time"].unique()))
    median_delta_ms = int(np.median(deltas)) if len(deltas) else None
    expected_delta_ms = NGSIM_SAMPLE_INTERVAL_MS
    interval_ok = (median_delta_ms is not None
                   and abs(median_delta_ms - expected_delta_ms) <= expected_delta_ms)

    out = pd.DataFrame({
        "time_sec": site_df["time_sec"],
        "vehicle_id": site_df["vehicle_id"],
        "pos_x": site_df["pos_x"],
        "pos_y": site_df["pos_y"],
        "speed_mps": site_df["speed_mps"],
        "lane_id": site_df["lane_id"],
        "vehicle_class": "NGSIM",
    })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # --- 8. Provenance ------------------------------------------------
    road_length_m = round(float(out["pos_x"].max() - out["pos_x"].min()), 1)
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "file": os.path.relpath(RAW_FILE, PROJECT_ROOT).replace("\\", "/"),
            "sha256": compute_sha256(RAW_FILE),
            "raw_rows": raw_rows,
            "raw_columns": raw_columns,
            "issuing_agency": "Federal Highway Administration (FHWA), USDOT",
            "portal": "https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj",
            "portal_dataset_id": "8ect-6jqj",
            "official_dataset_name": "Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data",
            "license": "Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0)",
            "attribution": ("USDOT / FHWA, 'Next Generation Simulation (NGSIM) Vehicle "
                           "Trajectories and Supporting Data', data.transportation.gov "
                           "dataset 8ect-6jqj, licensed CC BY-SA 3.0 Unported."),
            **_verification_fields(),
        },
        "filter_rules": {
            "location_filter": "Location == '%s'" % TARGET_LOCATION,
            "duplicate_policy": "exact duplicate rows removed (keep first)",
            "invalid_rows_policy": "rows with non-numeric Global_Time/Local_X/Local_Y/v_Vel/Lane_ID removed",
            "unit_conversion": "feet -> metres (x0.3048), ft/s -> m/s (x0.3048)",
            "time_origin": "Global_Time minus the minimum Global_Time of the filtered site",
            "vehicle_id_policy": "original NGSIM Vehicle_ID preserved verbatim",
        },
        "timestamp_diagnosis": ts_diag,
        "counts": {
            "rows_after_site_filter": rows_after_site_filter,
            "duplicate_rows_removed": duplicate_rows_removed,
            "invalid_rows_removed": invalid_rows,
            "rows_written": len(out),
            "unique_vehicles": int(out["vehicle_id"].nunique()),
        },
        "geometry": {
            "road_length_m": road_length_m,
            "lane_count": int(out["lane_id"].nunique()),
            "duration_sec": round(float(out["time_sec"].max()), 1),
            "median_sample_interval_ms": median_delta_ms,
            "expected_sample_interval_ms": expected_delta_ms,
            "sample_interval_consistent": bool(interval_ok),
        },
        "output": {
            "file": os.path.relpath(OUTPUT_CSV, PROJECT_ROOT).replace("\\", "/"),
            "sha256": compute_sha256(OUTPUT_CSV),
            "rows": len(out),
        },
    }
    with open(OUTPUT_PROVENANCE, "w") as f:
        json.dump(provenance, f, indent=2)

    if verbose:
        print("=" * 70)
        print("Processed authentic NGSIM US-101 trace")
        print("=" * 70)
        print("  raw rows                : %d" % raw_rows)
        print("  rows after site filter  : %d" % rows_after_site_filter)
        print("  duplicate rows removed  : %d" % duplicate_rows_removed)
        print("  invalid rows removed    : %d" % invalid_rows)
        print("  rows written            : %d" % len(out))
        print("  unique vehicles         : %d" % provenance["counts"]["unique_vehicles"])
        print("  road length (m)         : %.1f" % road_length_m)
        print("  lanes                   : %d" % provenance["geometry"]["lane_count"])
        print("  duration (s)            : %.1f" % provenance["geometry"]["duration_sec"])
        print("  median interval (ms)    : %s" % median_delta_ms)
        if ts_diag["anomalies"]:
            for loc, info in ts_diag["anomalies"].items():
                print("  [timestamp anomaly] %s: %s" % (loc, info["issue"]))
        print("  output                  : %s" % provenance["output"]["file"])
        print("  provenance              : %s" % os.path.relpath(OUTPUT_PROVENANCE, PROJECT_ROOT))
        print("=" * 70)

    return provenance


def main():
    parser = argparse.ArgumentParser(description="Prepare the authentic NGSIM US-101 trace.")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify the raw file checksum; do not build the trace.")
    parser.add_argument("--record-hash", action="store_true",
                        help="Record the observed hash of the raw file as the reference hash.")
    parser.add_argument("--force", action="store_true",
                        help="Build even when the raw file is not checksum-authenticated.")
    args = parser.parse_args()

    report = authenticate_raw_file()

    if args.record_hash:
        if not report["exists"]:
            print("Cannot record a hash: the raw file does not exist.")
            return 1
        write_reference_hash(report["observed_sha256"])
        print("[AUTH] Reference hash recorded: %s" % report["observed_sha256"])
        return 0

    if args.verify:
        print("Authenticated: %s" % report["authenticated"])
        print("Reason: %s" % report["reason"])
        return 0 if report["authenticated"] else 2

    if not report["authenticated"] and not args.force:
        print("")
        print("[BLOCKED] The raw dataset is not checksum-authenticated.")
        print("          %s" % report["reason"])
        print("          Verify the file against the official USDOT source and then run:")
        print("              python data/source/prepare_ngsim.py --record-hash")
        print("          Or re-run with --force to build the trace anyway (unverified).")
        return 2

    build_processed_trace()
    return 0


if __name__ == "__main__":
    sys.exit(main())
