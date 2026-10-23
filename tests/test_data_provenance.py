"""
Tests for the dataset provenance and licence statements.

These check that the repository does not claim more than it can prove:

* the raw export's authentication status is recorded with its evidence;
* the licence stated in the repository matches the official catalogue
  (CC BY-SA 3.0 Unported, not CC0 / public domain);
* the processed trace carries a transformation manifest and a checksum;
* the raw source file is never modified by the processing pipeline.
"""

import hashlib
import io
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "source", "us101_raw.csv")
RECORD = os.path.join(ROOT, "data", "source", "verification_record.json")
PROVENANCE = os.path.join(ROOT, "data", "traces", "ngsim_us101_authentic.provenance.json")
PROCESSED = os.path.join(ROOT, "data", "traces", "ngsim_us101_authentic.csv")
FINGERPRINT = os.path.join(ROOT, "data", "source", "us101_raw.sha256")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestSourceAuthentication(unittest.TestCase):

    @unittest.skipUnless(os.path.exists(RECORD), "no verification record present")
    def test_verification_record_is_complete(self):
        rec = json.loads(read(RECORD))
        for field in ("verification_attempted_utc", "source_url",
                      "retrieval_succeeded", "downloaded_sha256",
                      "local_sha256", "byte_identical",
                      "difference_detail", "official_license_field",
                      "official_dataset_name", "published_checksum_found",
                      "independent_authentication_status"):
            self.assertIn(field, rec, "the verification record is missing '%s'" % field)

    @unittest.skipUnless(os.path.exists(RECORD), "no verification record present")
    def test_no_published_checksum_is_claimed(self):
        rec = json.loads(read(RECORD))
        self.assertFalse(rec["published_checksum_found"],
                         "the catalogue publishes no checksum; none may be claimed")
        self.assertIsNone(rec["published_checksum_value"])
        self.assertIn("no checksum", rec["published_checksum_note"].lower()
                      if "published_checksum_note" in rec else "no checksum")

    @unittest.skipUnless(os.path.exists(RECORD), "no verification record present")
    def test_authentication_status_is_explicit(self):
        rec = json.loads(read(RECORD))
        status = rec["independent_authentication_status"]
        self.assertIn(status, ("authenticated_by_fresh_download_comparison",
                                "unresolved"))
        if status == "authenticated_by_fresh_download_comparison":
            # The claim must be backed by a real retrieval.
            self.assertTrue(rec["retrieval_succeeded"])
            self.assertTrue(rec["downloaded_sha256"])
            self.assertTrue(rec["difference_detail"])

    def test_fingerprint_file_does_not_claim_authentication(self):
        if not os.path.exists(FINGERPRINT):
            self.skipTest("no fingerprint file present")
        content = read(FINGERPRINT)
        lowered = content.lower()
        self.assertIn("reproducibility fingerprint", lowered)
        # The file must state, in some wording, that the hash is not an
        # authenticity proof and that the independent evidence lives elsewhere.
        self.assertIn("not an authenticity proof", lowered)
        self.assertIn("verification_record.json", content)
        self.assertIn("8ect-6jqj", content)
        # The recorded hash must be the one of the current raw file.
        with io.open(RAW, "rb") as f:
            h = hashlib.sha256()
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        self.assertIn(h.hexdigest(), content,
                      "the fingerprint does not match the installed raw file")

    def test_downloader_separates_fingerprint_from_authentication(self):
        src = read(os.path.join(ROOT, "data", "source", "download_ngsim.py"))
        self.assertIn("REPRODUCIBILITY FINGERPRINT", src)
        self.assertIn("INDEPENDENT AUTHENTICATION", src)
        self.assertIn("verification_record.json", src)


class TestLicenceStatements(unittest.TestCase):
    """The licence must match the official catalogue."""

    OFFICIAL = "CC BY-SA 3.0"

    def test_no_cc0_or_public_domain_claims_remain(self):
        offenders = []
        for rel in ("data/source/download_ngsim.py",
                     "data/source/prepare_ngsim.py",
                     "data/source/README.md",
                     "data/PROVENANCE.md"):
            path = os.path.join(ROOT, rel)
            if not os.path.exists(path):
                continue
            text = read(path)
            # Ignore the *historical correction note* that explains the old
            # claim and explicit negations such as "not CC0": only a live
            # claim is a problem.
            for line in text.splitlines():
                if ("CC0" in line or "Public Domain" in line
                        or "17 U.S.C" in line):
                    lowered = line.lower()
                    if ("corrected" in lowered or "does not support" in lowered
                            or "not cc0" in lowered or "no longer" in lowered
                            or "rather than cc0" in lowered
                            or "instead of cc0" in lowered):
                        continue
                    offenders.append("%s: %s" % (rel, line.strip()))
        self.assertEqual(offenders, [],
                         "the repository still claims a licence the official "
                         "catalogue does not list:\n  " + "\n  ".join(offenders))

    def test_official_licence_is_stated(self):
        for rel in ("data/source/download_ngsim.py",
                     "data/source/prepare_ngsim.py",
                     "data/source/README.md",
                     "data/PROVENANCE.md"):
            path = os.path.join(ROOT, rel)
            if not os.path.exists(path):
                continue
            self.assertIn("Share Alike 3.0", read(path),
                          "%s does not state the official licence" % rel)

    def test_attribution_is_present(self):
        src = read(os.path.join(ROOT, "data", "source", "README.md"))
        self.assertIn("8ect-6jqj", src)
        self.assertRegex(src, r"Attribution")


class TestProcessedTraceProvenance(unittest.TestCase):

    @unittest.skipUnless(os.path.exists(PROVENANCE), "no processed-trace provenance")
    def test_provenance_has_a_transformation_manifest(self):
        prov = json.loads(read(PROVENANCE))
        self.assertIn("filter_rules", prov)
        rules = prov["filter_rules"]
        for key in ("location_filter", "duplicate_policy",
                    "invalid_rows_policy", "unit_conversion",
                    "time_origin", "vehicle_id_policy"):
            self.assertIn(key, rules)
        self.assertIn("us-101", rules["location_filter"])

    @unittest.skipUnless(os.path.exists(PROVENANCE), "no processed-trace provenance")
    def test_provenance_records_the_output_checksum(self):
        prov = json.loads(read(PROVENANCE))
        self.assertIn("output", prov)
        self.assertEqual(len(prov["output"]["sha256"]), 64)
        if os.path.exists(PROCESSED):
            from experiments.experiment_runner import get_file_sha256
            self.assertEqual(prov["output"]["sha256"], get_file_sha256(PROCESSED),
                             "the processed file does not match its recorded checksum")

    @unittest.skipUnless(os.path.exists(PROVENANCE), "no processed-trace provenance")
    def test_provenance_states_the_authentication_status(self):
        prov = json.loads(read(PROVENANCE))
        source = prov.get("source", {})
        self.assertIn("independent_authentication_status", source)
        self.assertIn("license", source)
        self.assertIn("Share Alike 3.0", source["license"])

    @unittest.skipUnless(os.path.exists(RAW), "the raw export is not present")
    def test_processing_does_not_modify_the_raw_source(self):
        from data.source import prepare_ngsim
        before = prepare_ngsim.compute_sha256(RAW)
        prepare_ngsim.build_processed_trace(verbose=False)
        after = prepare_ngsim.compute_sha256(RAW)
        self.assertEqual(before, after, "the raw source file was modified")


if __name__ == "__main__":
    unittest.main()
