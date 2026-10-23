# Authentic Source Data Reference & Download Instructions

This directory is designated exclusively for **genuine, raw source data downloads** to ensure that authentic empirical files are never conflated with or overwritten by synthetic trajectory generators.

---

## 1. FHWA NGSIM Raw Export Download

To obtain the authentic Federal Highway Administration (FHWA) NGSIM raw export:

1. **Official USDOT Portal:**
   Visit: `https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj`
2. **Download Target:**
   `us-101-trajectories-0750am-0805am.csv`
3. **Save Location:**
   `data/source/us101_raw.csv`

### Important: the raw export is NOT a single road trace

The official export aggregates **four road sites** and therefore cannot be
replayed directly:

| Site | Rows | Notes |
|---|---|---|
| `us-101` | 211,721 | The US-101 study segment (~675 m, 8 lanes) |
| `i-80` | 79,268 | I-80 freeway segment |
| `lankershim` | 66,051 | Lankershim Boulevard arterial |
| `peachtree` | 35,920 | Peachtree Street arterial |

Additional properties of the raw file (measured by `prepare_ngsim.py`):

* **392,961 rows, 25 columns** (not 18 as sometimes documented).
* **28,954 exact duplicate rows**, all inside the `us-101` records.
* **Timestamp-scale anomaly:** Peachtree records use a ~1.16×10⁹ epoch while
  the other sites use ~1.12×10¹². The two scales are not comparable.
* **Sampling interval:** 0.1 s (100 ms), verified by the median inter-record
  delta.
* **US-101 geometry:** `Local_Y` spans 1.4 m – 675.9 m, i.e. a ~675 m study
  segment, **not** the multi-kilometre highway assumed by the default
  deployment.

### License
* **Agency:** United States Department of Transportation (USDOT) / Federal Highway Administration (FHWA)
* **Licence (as listed in the official USDOT Socrata catalogue):** Creative Commons
  Attribution | Share Alike 3.0 Unported (`CC_30_BY_SA`).
  The official catalogue metadata for dataset `8ect-6jqj` carries this licence
  field; it was verified directly from the portal metadata JSON on
  2026-10-21 (see `verification_record.json`).  Earlier versions of this file
  stated CC0 1.0 / U.S. Public Domain, which the catalogue does not support,
  so that claim has been corrected.
* **Required attribution:** USDOT / FHWA, *Next Generation Simulation (NGSIM)
  Vehicle Trajectories and Supporting Data*, data.transportation.gov dataset
  `8ect-6jqj`, licensed CC BY-SA 3.0 Unported.
* **Note on redistribution:** CC BY-SA 3.0 requires that redistributed
  material carries the same licence and an attribution notice.

---

## 2. Integrity Verification (SHA-256)

The raw file is authenticated by checksum. The reference hash is stored in
`us101_raw.sha256`.

```bash
# Verify the raw export
python data/source/prepare_ngsim.py --verify

# Record a new reference hash (only after checking the file against the
# official USDOT source yourself)
python data/source/download_ngsim.py --record-hash
```

A file with **no** recorded reference hash is reported as *unauthenticated* and
is never treated as verified. The downloader will not report success for an
unverified file.

---

## 3. Processing Pipeline

`prepare_ngsim.py` converts the raw export into a replay-ready metric trace:

```bash
python data/source/prepare_ngsim.py
```

The pipeline:

1. authenticates the raw file against `us101_raw.sha256`,
2. diagnoses and reports timestamp-scale anomalies per road site,
3. filters to the intended site (`Location == 'us-101'`),
4. removes exact duplicate records,
5. drops rows with unusable numeric values,
6. preserves the original source `Vehicle_ID` values,
7. converts feet → metres and ft/s → m/s,
8. re-bases the time origin to the filtered site,
9. writes `data/traces/ngsim_us101_authentic.csv` and a provenance sidecar
   `data/traces/ngsim_us101_authentic.provenance.json` containing the source
   hash, the filter rules, the row/duplicate/anomaly counts, the derived
   geometry and the output checksum.

The raw file is never modified.

### Using the processed trace

Select **"Authentic FHWA NGSIM US-101 Trace"** in the web console's
*Mobility & Traces* panel, or use it programmatically:

```python
env = VANETOffloadingEnv(mobility_mode="authentic_trace")
print(env.trace_geometry)   # road length, lane count, vehicle count, duration
print([rsu.x for rsu in env.rsus])  # RSUs placed inside the observed segment
```

The road length, lane count and RSU positions are derived from the observed
trajectory extent, so no RSU ends up outside the replayed road segment.

---

## 4. Raw Data Schema Reference

The raw export contains 25 columns per trajectory row:

* `Vehicle_ID`: unique integer vehicle identifier.
* `Frame_ID`: video frame counter.
* `Total_Frames`: total observed frames.
* `Global_Time`: epoch timestamp in **milliseconds** (see the Peachtree anomaly).
* `Local_X`: lateral coordinate from the left pavement edge (feet).
* `Local_Y`: longitudinal coordinate from the freeway entrance (feet).
* `Global_X` / `Global_Y`: State Plane coordinates (feet).
* `v_Length` / `v_Width`: vehicle dimensions (feet).
* `v_Class`: vehicle category (1 = Motorcycle, 2 = Automobile, 3 = Heavy Truck).
* `v_Vel`: instantaneous velocity (ft/s).
* `v_Acc`: instantaneous acceleration (ft/s²).
* `Lane_ID`: travel lane (1 = leftmost HOV/passing lane).
* `O_Zone` / `D_Zone`: origin/destination zone identifiers.
* `Int_ID` / `Section_ID`: intersection and section identifiers.
* `Direction`: direction of travel.
* `Movement`: through / left / right movement.
* `Preceding` / `Following`: leading and following vehicle IDs.
* `Space_Headway`: space headway (ft).
* `Time_Headway`: time headway (s).
* `Location`: road site identifier (`us-101`, `i-80`, `lankershim`, `peachtree`).

---

## 5. Files in this directory

| File | Purpose |
|---|---|
| `us101_raw.csv` | Immutable authentic raw export. Never modified. |
| `us101_raw.sha256` | Reference SHA-256 used for authentication. |
| `download_ngsim.py` | Download + checksum authentication. |
| `prepare_ngsim.py` | Validation, filtering and processed-trace builder. |
| `README.md` | This document. |
