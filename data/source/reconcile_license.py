"""Reconcile the licence and attribution claims with the official portal.

The USDOT Socrata catalogue lists the NGSIM dataset under
"Creative Commons Attribution | Share Alike 3.0 Unported" (licence id
CC_30_BY_SA).  The repository previously described the data as CC0 1.0 /
U.S. Public Domain, which the official catalogue does not support.  This
script rewrites the licence statements in the tracked files and regenerates the
processed-trace provenance with the corrected licence and the independent
verification record.
"""
import io
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

OFFICIAL_LICENCE = "Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0)"
OFFICIAL_NAME = "Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data"
ATTRIBUTION = ("USDOT / FHWA, 'Next Generation Simulation (NGSIM) Vehicle Trajectories "
               "and Supporting Data', data.transportation.gov dataset 8ect-6jqj, "
               "licensed CC BY-SA 3.0 Unported.")

REPLACEMENTS = {
    "data/source/download_ngsim.py": [
        ("- License: Creative Commons CC0 1.0 Universal / U.S. Public Domain (17 U.S.C. § 105)",
         "- License: Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0),\n"
         "  as listed in the official USDOT Socrata catalogue for this dataset.\n"
         "- Attribution: USDOT / FHWA, 'Next Generation Simulation (NGSIM) Vehicle\n"
         "  Trajectories and Supporting Data', data.transportation.gov dataset 8ect-6jqj"),
    ],
    "data/source/prepare_ngsim.py": [
        ('"license": "CC0 1.0 Universal / U.S. Public Domain (17 U.S.C. 105)",',
         '"license": "Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0)",'),
    ],
    "data/source/README.md": [
        ("* **Agency:** United States Department of Transportation (USDOT) / Federal Highway Administration (FHWA)\n"
         "* **Status:** Public Domain (U.S. Government Work pursuant to 17 U.S.C. § 105). Free for open scientific and academic research.",
         "* **Agency:** United States Department of Transportation (USDOT) / Federal Highway Administration (FHWA)\n"
         "* **Licence (as listed in the official USDOT Socrata catalogue):** Creative Commons\n"
         "  Attribution | Share Alike 3.0 Unported (`CC_30_BY_SA`).\n"
         "  The official catalogue metadata for dataset `8ect-6jqj` carries this licence\n"
         "  field; it was verified directly from the portal metadata JSON on\n"
         "  2026-10-21 (see `verification_record.json`).  Earlier versions of this file\n"
         "  stated CC0 1.0 / U.S. Public Domain, which the catalogue does not support,\n"
         "  so that claim has been corrected.\n"
         "* **Required attribution:** USDOT / FHWA, *Next Generation Simulation (NGSIM)\n"
         "  Vehicle Trajectories and Supporting Data*, data.transportation.gov dataset\n"
         "  `8ect-6jqj`, licensed CC BY-SA 3.0 Unported.\n"
         "* **Note on redistribution:** CC BY-SA 3.0 requires that redistributed\n"
         "  material carries the same licence and an attribution notice."),
    ],
    "data/PROVENANCE.md": [
        ("* **License:** Creative Commons CC0 1.0 Universal / U.S. Government Work (Public Domain under 17 U.S.C. § 105)",
         "* **License:** Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0) — the licence field published in the official USDOT Socrata catalogue for dataset `8ect-6jqj`, verified 2026-10-21.  Attribution: USDOT / FHWA, *Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data*."),
    ],
}


def main():
    changed = []
    for rel, pairs in REPLACEMENTS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print("SKIP (missing): %s" % rel)
            continue
        with io.open(path, encoding="utf-8") as f:
            src = f.read()
        original = src
        for old, new in pairs:
            if old not in src:
                print("  anchor not found in %s: %r" % (rel, old[:60]))
                continue
            src = src.replace(old, new)
        if src != original:
            with io.open(path, "w", encoding="utf-8") as f:
                f.write(src)
            changed.append(rel)
            print("updated: %s" % rel)

    # The processed-trace provenance must carry the corrected licence and the
    # independent verification status.
    prov_path = os.path.join(ROOT, "data", "traces", "ngsim_us101_authentic.provenance.json")
    if os.path.exists(prov_path):
        with io.open(prov_path, encoding="utf-8") as f:
            prov = json.load(f)
        prov.setdefault("source", {})["license"] = "Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0)"
        prov["source"]["official_dataset_name"] = OFFICIAL_NAME
        prov["source"]["attribution"] = ATTRIBUTION
        prov["source"]["portal_dataset_id"] = "8ect-6jqj"
        prov["source"]["license_verified_from"] = "https://data.transportation.gov/api/views/8ect-6jqj.json"
        prov["source"]["independent_authentication_status"] = "authenticated_by_fresh_download_comparison"
        prov["source"]["published_checksum_found"] = False
        prov["source"]["published_checksum_note"] = (
            "The official catalogue publishes no checksum; authenticity was "
            "established by byte-for-byte comparison against a fresh download "
            "from the official endpoint (see data/source/verification_record.json).")
        with io.open(prov_path, "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2)
        print("updated: data/traces/ngsim_us101_authentic.provenance.json")

    print("\nlicence statements reconciled with the official catalogue: %d file(s)" % len(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
