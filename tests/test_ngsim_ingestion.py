"""
Tests for the authentic FHWA NGSIM US-101 ingestion pipeline.

These verify that:

* the raw export is authenticated by checksum (and that a missing reference
  hash is reported as *unauthenticated* rather than silently accepted),
* the pipeline filters to the intended road site,
* exact duplicate records are removed,
* the timestamp-scale anomaly for the Peachtree records is detected,
* source vehicle IDs are preserved,
* the processed trace replays positions that match the known source records.
"""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from data.source import prepare_ngsim as p


RAW_PRESENT = os.path.exists(p.RAW_FILE)
PROCESSED_PRESENT = os.path.exists(p.OUTPUT_CSV)


class TestRawAuthentication(unittest.TestCase):
    """The raw export must be authenticated by checksum."""

    def test_authenticate_reports_observed_hash(self):
        report = p.authenticate_raw_file(verbose=False)
        self.assertIn("observed_sha256", report)
        self.assertIn("authenticated", report)
        self.assertIn("reason", report)
        if RAW_PRESENT:
            self.assertEqual(len(report["observed_sha256"]), 64)

    def test_missing_reference_hash_is_not_authenticated(self):
        """A file without a recorded reference hash must not count as verified."""
        if not RAW_PRESENT:
            self.skipTest("Raw NGSIM export is not present.")

        original = p.read_reference_hash
        try:
            p.read_reference_hash = lambda: ""
            report = p.authenticate_raw_file(verbose=False)
            self.assertFalse(report["authenticated"])
            self.assertIn("reference", report["reason"].lower())
        finally:
            p.read_reference_hash = original

    def test_mismatching_hash_is_rejected(self):
        if not RAW_PRESENT:
            self.skipTest("Raw NGSIM export is not present.")

        original = p.read_reference_hash
        try:
            p.read_reference_hash = lambda: "0" * 64
            report = p.authenticate_raw_file(verbose=False)
            self.assertFalse(report["authenticated"])
            self.assertIn("mismatch", report["reason"].lower())
        finally:
            p.read_reference_hash = original

    def test_downloader_rejects_unauthenticated_file(self):
        """download_ngsim must not report success for an unverified file."""
        from data.source import download_ngsim as d

        if not os.path.exists(d.DEST_FILE):
            self.skipTest("Raw NGSIM export is not present.")

        original = d.read_reference_hash
        try:
            d.read_reference_hash = lambda: ""
            ok, digest = d.verify_or_download_ngsim(force_download=False)
            self.assertFalse(ok, "An unverified file must not be reported as successful")
            self.assertIsNotNone(digest, "The observed hash should still be reported")
        finally:
            d.read_reference_hash = original


class TestIngestionPipeline(unittest.TestCase):
    """The processed trace must be a faithful, validated subset of the raw export."""

    @classmethod
    def setUpClass(cls):
        if not RAW_PRESENT:
            raise unittest.SkipTest("Raw NGSIM export is not present.")
        cls.raw = pd.read_csv(p.RAW_FILE)
        if not PROCESSED_PRESENT:
            p.build_processed_trace(verbose=False)
        with open(p.OUTPUT_PROVENANCE) as f:
            cls.provenance = json.load(f)
        cls.processed = pd.read_csv(p.OUTPUT_CSV)

    def test_filters_to_intended_road_site(self):
        counts = self.provenance["counts"]
        raw_rows = self.provenance["source"]["raw_rows"]
        self.assertLess(counts["rows_after_site_filter"], raw_rows)
        self.assertGreater(counts["rows_after_site_filter"], 0)

    def test_duplicate_rows_removed(self):
        counts = self.provenance["counts"]
        self.assertGreater(counts["duplicate_rows_removed"], 0)
        # The processed file must contain no duplicated rows.
        self.assertEqual(int(self.processed.duplicated().sum()), 0)

    def test_timestamp_anomaly_detected(self):
        diag = self.provenance["timestamp_diagnosis"]
        self.assertIn("anomalies", diag)
        # The Peachtree records use a different epoch scale and must be flagged.
        self.assertIn("peachtree", diag["anomalies"])
        issue = diag["anomalies"]["peachtree"]["issue"]
        self.assertIn("timestamp scale", issue)

    def test_sample_interval_consistent(self):
        geom = self.provenance["geometry"]
        self.assertEqual(geom["expected_sample_interval_ms"], 100)
        self.assertTrue(geom["sample_interval_consistent"])
        self.assertEqual(geom["median_sample_interval_ms"], 100)

    def test_source_vehicle_ids_preserved(self):
        raw_us = self.raw[self.raw["Location"].astype(str).str.lower() == "us-101"]
        raw_ids = set(raw_us["Vehicle_ID"].astype(int).unique())
        processed_ids = set(self.processed["vehicle_id"].unique())
        self.assertTrue(processed_ids.issubset(raw_ids),
                        "Processed vehicle IDs must come from the source export")
        self.assertGreater(len(processed_ids), 100)

    def test_geometry_matches_known_segment(self):
        """The US-101 study segment is roughly 640-700 m long."""
        geom = self.provenance["geometry"]
        self.assertGreater(geom["road_length_m"], 600.0)
        self.assertLess(geom["road_length_m"], 750.0)
        self.assertEqual(geom["lane_count"], 8)

    def test_replay_matches_known_source_records(self):
        """Positions in the processed trace must match the raw source records."""
        raw_us = self.raw[self.raw["Location"].astype(str).str.lower() == "us-101"].copy()
        raw_us = raw_us.drop_duplicates()
        raw_us["Local_X"] = pd.to_numeric(raw_us["Local_X"], errors="coerce")
        raw_us["Local_Y"] = pd.to_numeric(raw_us["Local_Y"], errors="coerce")
        raw_us["v_Vel"] = pd.to_numeric(raw_us["v_Vel"], errors="coerce")
        raw_us = raw_us.dropna(subset=["Local_X", "Local_Y", "v_Vel"])
        raw_us["vehicle_id"] = raw_us["Vehicle_ID"].astype(int)
        raw_us["pos_x"] = (raw_us["Local_Y"] * p.FEET_TO_M).round(2)
        raw_us["pos_y"] = (raw_us["Local_X"] * p.FEET_TO_M).round(2)
        raw_us["speed_mps"] = (raw_us["v_Vel"] * p.FEET_TO_M).round(2)
        t_min = raw_us["Global_Time"].min()
        raw_us["time_sec"] = ((raw_us["Global_Time"] - t_min) / 1000.0).round(1)

        # Pick a few well-populated source vehicles and verify their tracks.
        popular = raw_us["vehicle_id"].value_counts().head(5).index.tolist()
        self.assertTrue(popular)

        for vid in popular:
            src = raw_us[raw_us["vehicle_id"] == vid].sort_values("time_sec")
            got = self.processed[self.processed["vehicle_id"] == vid].sort_values("time_sec")
            self.assertGreater(len(got), 0, f"vehicle {vid} missing from processed trace")
            # Compare a sample of records.
            for idx in range(0, min(len(src), 20), 5):
                s_row = src.iloc[idx]
                g_row = got.iloc[idx]
                self.assertAlmostEqual(float(s_row["pos_x"]), float(g_row["pos_x"]), places=1)
                self.assertAlmostEqual(float(s_row["pos_y"]), float(g_row["pos_y"]), places=1)
                self.assertAlmostEqual(float(s_row["speed_mps"]), float(g_row["speed_mps"]), places=1)
                self.assertAlmostEqual(float(s_row["time_sec"]), float(g_row["time_sec"]), places=1)

    def test_provenance_records_source_checksum(self):
        self.assertEqual(len(self.provenance["source"]["sha256"]), 64)
        self.assertEqual(self.provenance["source"]["sha256"], p.compute_sha256(p.RAW_FILE))
        self.assertIn("filter_rules", self.provenance)
        self.assertIn("us-101", self.provenance["filter_rules"]["location_filter"])

    def test_processed_output_checksum_recorded(self):
        self.assertEqual(len(self.provenance["output"]["sha256"]), 64)
        self.assertEqual(self.provenance["output"]["sha256"], p.compute_sha256(p.OUTPUT_CSV))


class TestTraceReplayIntegration(unittest.TestCase):
    """The environment must replay the authentic trace with matched geometry."""

    def test_authentic_trace_mode_geometry(self):
        if not PROCESSED_PRESENT:
            self.skipTest("Processed authentic trace is not present.")

        from environment.vanet_env import VANETOffloadingEnv

        env = VANETOffloadingEnv(num_vehicles=6, attack_ratio=0.0, mobility_mode="authentic_trace")

        self.assertIn("authentic", env.trace_file)
        self.assertIsNotNone(env.trace_geometry)
        # The road length must match the observed segment, not the default
        # multi-kilometre highway.
        self.assertLess(env.road_length, 1000.0)
        self.assertEqual(env.num_lanes, env.trace_geometry["lane_count"])

        # Every RSU must sit inside the observed road segment.
        geom = env.trace_geometry
        for rsu in env.rsus:
            self.assertGreaterEqual(rsu.x, geom["x_min_m"] - 1.0)
            self.assertLessEqual(rsu.x, geom["x_max_m"] + 1.0)

        # Vehicles must stay inside the segment while replaying.
        for _ in range(4):
            env.update_positions()
        for v in env.vehicles.values():
            self.assertGreaterEqual(v.x, 0.0)
            self.assertLessEqual(v.x, env.road_length)


if __name__ == "__main__":
    unittest.main()
