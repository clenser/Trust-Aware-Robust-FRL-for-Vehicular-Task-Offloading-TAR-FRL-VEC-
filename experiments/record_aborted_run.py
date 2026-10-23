"""Record the empty aborted run directory in the run manifest.

The directory `results/run_20261022_005703_corrected_full` was created by an
aborted benchmark invocation and contains no artefacts.  It is recorded so
that every directory under `results/` is accounted for in the manifest.
"""
import io
import json
import os

RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
TARGETS = ("run_20261022_005703_corrected_full",
           "run_20261022_140055_corrected_participation_rule")

path = os.path.join(RESULTS, "manifest.json")
with io.open(path, encoding="utf-8") as f:
    manifest = json.load(f)

added = []
for target in TARGETS:
    if any(e.get("run_id") == target for e in manifest):
        continue
    if not os.path.isdir(os.path.join(RESULTS, target)):
        continue
    manifest.append({
        "run_id": target,
        "directory": os.path.join("results", target),
        "timestamp_utc": None,
        "num_seeds": 0,
        "seeds": [],
        "validation_status": "aborted",
        "published_as_active": False,
        "note": "Aborted benchmark invocation. The directory contains no "
                "artefacts (no results, no metadata, no figures); it is kept "
                "for completeness and must not be cited.",
    })
    added.append(target)

if added:
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("recorded %s in the manifest" % ", ".join(added))
else:
    print("nothing to record")
