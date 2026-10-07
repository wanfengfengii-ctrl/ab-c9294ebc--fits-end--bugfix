"""Unit tests for checksum materialization (fitsaudit.core)."""

import os
import struct
import unittest

from fitsaudit import fixtures
from fitsaudit.core import (
    ACCEPTED,
    REJECTED,
    MaterializationFailure,
    audit_bytes,
    char_encode,
    materialize_checksums,
    sum32_be,
    UINT32_MAX,
)
from fitsaudit.fixtures import (
    card,
    data_block,
    header_block,
    image_hdu,
    primary_hdu,
)
from verify import stdfits


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _verdicts(report):
    return [(h["datasum"]["verdict"], h["checksum"]["verdict"])
            for h in report["hdus"]]


def _assert_std_reader_accepts(testcase, blob, hdu_count=None):
    """Strict END-stopping reader sees valid checksum cards before END."""
    ok, hdus = stdfits.verify_materialized(blob)
    testcase.assertTrue(ok, hdus)
    if hdu_count is not None:
        testcase.assertEqual(len(hdus), hdu_count)
    for hdu in hdus:
        testcase.assertTrue(hdu["datasum"]["present"])
        testcase.assertTrue(hdu["datasum"]["valid"])
        testcase.assertTrue(hdu["datasum"]["card_before_end"])
        testcase.assertTrue(hdu["checksum"]["present"])
        testcase.assertTrue(hdu["checksum"]["valid"])
        testcase.assertTrue(hdu["checksum"]["card_before_end"])
    return hdus


class MaterializeSuccessTests(unittest.TestCase):
    def test_missing_cards_are_filled(self):
        early = fixtures.build_valid_file(2, checksums=False)
        before = audit_bytes(early)
        self.assertEqual(before["conclusion"], ACCEPTED)
        self.assertEqual(_verdicts(before),
                         [("ABSENT", "ABSENT")] * 3)

        completed = materialize_checksums(early)
        after = audit_bytes(completed)
        self.assertEqual(after["conclusion"], ACCEPTED)
        self.assertEqual(_verdicts(after), [("VALID", "VALID")] * 3)

    def test_size_and_hdu_boundaries_unchanged(self):
        early = fixtures.build_valid_file(2, checksums=False)
        completed = materialize_checksums(early)
        self.assertEqual(len(completed), len(early))
        before = audit_bytes(early)
        after = audit_bytes(completed)
        for b, a in zip(before["hdus"], after["hdus"]):
            self.assertEqual(b["range"], a["range"])
            self.assertEqual(b["header"]["offset"], a["header"]["offset"])
            self.assertEqual(b["header"]["bytes"], a["header"]["bytes"])
            self.assertEqual(b["data"], a["data"])

    def test_scientific_payload_byte_identical(self):
        early = fixtures.build_valid_file(2, checksums=False)
        completed = materialize_checksums(early)
        before = audit_bytes(early)
        for h in before["hdus"]:
            start = h["data"]["offset"]
            end = start + h["data"]["paddedBytes"]
            self.assertEqual(completed[start:end], early[start:end])
        # The only input bytes allowed to change are blank (space)
        # padding slots -- overwritten by a new card or by the shifted
        # END card -- and the old END card slots, which the inserted
        # cards overwrite (the END card itself moves to a former blank
        # slot).  Nothing else in the header may move or change.
        end_card_spans = []
        for h in before["hdus"]:
            # END is the last content card: one card before content end.
            content_bytes = (h["header"]["offset"]
                             + (h["header"]["cards"] - 1) * 80)
            end_card_spans.append((content_bytes, content_bytes + 80))

        def in_old_end_card(i):
            return any(lo <= i < hi for lo, hi in end_card_spans)

        for i, (a, b) in enumerate(zip(early, completed)):
            if a != b:
                self.assertTrue(
                    a == 0x20 or in_old_end_card(i),
                    "byte %d outside a blank slot/old END card changed: "
                    "0x%02X -> 0x%02X" % (i, a, b))

    def test_cards_land_directly_before_end(self):
        early = primary_hdu(checksums=False)
        old_end = fixtures.find_card(early, "END")
        completed = materialize_checksums(early)
        # Two cards inserted: the END card shifts exactly two slots
        # later, with DATASUM then CHECKSUM directly in front of it.
        new_end = old_end + 160
        self.assertEqual(completed[new_end:new_end + 80],
                         fixtures.END_CARD)
        self.assertEqual(completed[new_end - 160:new_end - 152],
                         b"DATASUM ")
        self.assertEqual(completed[new_end - 80:new_end - 72], b"CHECKSUM")
        # Everything after the (shifted) END stays pure space padding.
        self.assertEqual(completed[new_end + 80:2880], b" " * (2880 - new_end - 80))

    def test_datasum_value_matches_data(self):
        data = struct.pack(">12h", *range(-6, 6))
        early = primary_hdu(data, bitpix=16, axes=(12,), checksums=False)
        completed = materialize_checksums(early)
        report = audit_bytes(completed)
        hdu = report["hdus"][0]
        self.assertEqual(hdu["datasum"]["computed"], sum32_be(data))
        self.assertEqual(hdu["datasum"]["stored"], sum32_be(data))

    def test_exactly_two_free_slots_fit(self):
        # 33 content cards + END occupy 34 cards, leaving exactly 2 slots.
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        cards += [card("COMMENT", None) for _ in range(30)]
        early = header_block(cards)
        completed = materialize_checksums(early)
        self.assertEqual(len(completed), len(early))
        self.assertEqual(audit_bytes(completed)["conclusion"], ACCEPTED)

    def test_multiblock_header_uses_slots_before_end(self):
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        cards += [card("COMMENT", None) for _ in range(40)]  # 2 header blocks
        early = header_block(cards)
        self.assertEqual(len(early), 5760)
        old_end = fixtures.find_card(early, "END")
        completed = materialize_checksums(early)
        # END shifts by two slots but the header block count (and so the
        # data offset and file length) is unchanged.
        new_end = fixtures.find_card(completed, "END")
        self.assertEqual(new_end, old_end + 160)
        self.assertEqual(completed[new_end - 160:new_end - 152],
                         b"DATASUM ")
        self.assertEqual(completed[new_end - 80:new_end - 72], b"CHECKSUM")
        self.assertEqual(completed[new_end + 80:5760],
                         b" " * (5760 - new_end - 80))
        self.assertEqual(len(completed), len(early))
        self.assertEqual(audit_bytes(completed)["conclusion"], ACCEPTED)

    def test_mixed_hdus_only_missing_ones_change(self):
        data = struct.pack(">6h", *range(6))
        early = primary_hdu(checksums=True) + image_hdu(
            data, bitpix=16, axes=(6, 1), checksums=False)
        completed = materialize_checksums(early)
        # The already-complete primary HDU is byte-for-byte identical.
        self.assertEqual(completed[:2880], early[:2880])
        report = audit_bytes(completed)
        self.assertEqual(report["conclusion"], ACCEPTED)
        self.assertEqual(_verdicts(report),
                         [("VALID", "VALID"), ("VALID", "VALID")])

    def test_datasum_only_extension_keeps_card_adds_checksum(self):
        data = struct.pack(">6h", *range(6))
        datasum = sum32_be(data)
        cards = [card("XTENSION", "IMAGE"), card("BITPIX", 16),
                 card("NAXIS", 1), card("NAXIS1", 6), card("PCOUNT", 0),
                 card("GCOUNT", 1), card("DATASUM", datasum)]
        early = primary_hdu(checksums=True) + header_block(cards) \
            + data_block(data)
        self.assertEqual(audit_bytes(early)["conclusion"], ACCEPTED)
        completed = materialize_checksums(early)
        # Existing DATASUM card is preserved in place.
        old_off = fixtures.find_card(early, "DATASUM", occurrence=1)
        self.assertEqual(completed[old_off:old_off + 80],
                         early[old_off:old_off + 80])
        # The new CHECKSUM sits directly before the shifted END, i.e. in
        # the old END card's slot.
        old_end = fixtures.find_card(early, "END", occurrence=1)
        end_off = fixtures.find_card(completed, "END", occurrence=1)
        self.assertEqual(end_off, old_end + 80)
        self.assertEqual(completed[end_off - 80:end_off - 72], b"CHECKSUM")
        self.assertEqual(completed[end_off + 80:end_off + 88], b"        ")
        self.assertEqual(audit_bytes(completed)["conclusion"], ACCEPTED)

    def test_checksum_only_extension_recomputed_in_place(self):
        data = struct.pack(">6h", *range(6))
        datasum = sum32_be(data)
        cards = [card("XTENSION", "IMAGE"), card("BITPIX", 16),
                 card("NAXIS", 1), card("NAXIS1", 6), card("PCOUNT", 0),
                 card("GCOUNT", 1), card("CHECKSUM", "0" * 16)]
        # Make the CHECKSUM valid for the header without a DATASUM card.
        total = sum32_be(header_block(cards), datasum)
        cards[-1] = card("CHECKSUM", char_encode(~total & UINT32_MAX))
        early = primary_hdu(checksums=True) + header_block(cards) \
            + data_block(data)
        self.assertEqual(audit_bytes(early)["conclusion"], ACCEPTED)
        old_off = fixtures.find_card(early, "CHECKSUM", occurrence=1)

        completed = materialize_checksums(early)
        # CHECKSUM card slot does not move; only its value is recomputed.
        new_off = fixtures.find_card(completed, "CHECKSUM", occurrence=1)
        self.assertEqual(old_off, new_off)
        self.assertNotEqual(completed[old_off + 11:old_off + 27],
                            early[old_off + 11:old_off + 27])
        # DATASUM fills the old END slot; END shifts by exactly one slot.
        old_end = fixtures.find_card(early, "END", occurrence=1)
        end_off = fixtures.find_card(completed, "END", occurrence=1)
        self.assertEqual(end_off, old_end + 80)
        self.assertEqual(completed[end_off - 80:end_off - 72], b"DATASUM ")
        self.assertEqual(audit_bytes(completed)["conclusion"], ACCEPTED)


class IdempotencyTests(unittest.TestCase):
    def test_fully_checksummed_input_returned_unchanged(self):
        for blob in (fixtures.build_valid_file(2),
                     fixtures.build_max_hdus_file(),
                     open("tests/data/valid_astropy.fits", "rb").read()):
            self.assertEqual(materialize_checksums(blob), blob)

    def test_repeated_materialization_is_fixed_point(self):
        early = fixtures.build_valid_file(2, checksums=False)
        once = materialize_checksums(early)
        twice = materialize_checksums(once)
        self.assertEqual(once, twice)
        self.assertEqual(
            audit_bytes(once)["conclusion"], ACCEPTED)

    def test_idempotent_retry_after_partial_client_failure(self):
        # Simulating a client retry: posting the completed file again
        # must produce exactly the same bytes.
        early = fixtures.build_valid_file(1, checksums=False)
        first = materialize_checksums(early)
        for _ in range(3):
            first = materialize_checksums(first)
        self.assertEqual(first, materialize_checksums(early))


class InsufficientSpaceTests(unittest.TestCase):
    def _tight_primary(self, comments):
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        cards += [card("COMMENT", None) for _ in range(comments)]
        return header_block(cards)

    def test_one_free_slot_reports_insufficient_space(self):
        # 34 content cards + END = 35 cards -> 1 blank slot, needs 2.
        early = self._tight_primary(31)
        with self.assertRaises(MaterializationFailure) as ctx:
            materialize_checksums(early)
        failure = ctx.exception.report["failure"]
        self.assertEqual(failure["hdu"], 0)
        self.assertEqual(failure["reason"],
                         "INSUFFICIENT_HEADER_SPACE")
        self.assertEqual(failure["offset"], 34 * 80)  # END card offset
        self.assertEqual(failure["details"]["endOffset"], 34 * 80)
        self.assertEqual(failure["details"]["availableSlots"], 1)
        self.assertEqual(failure["details"]["requiredSlots"], 2)

    def test_zero_free_slots_reports_insufficient_space(self):
        early = self._tight_primary(32)  # header fills the block exactly
        with self.assertRaises(MaterializationFailure) as ctx:
            materialize_checksums(early)
        failure = ctx.exception.report["failure"]
        self.assertEqual(failure["reason"],
                         "INSUFFICIENT_HEADER_SPACE")
        self.assertEqual(failure["details"]["availableSlots"], 0)
        self.assertEqual(failure["details"]["requiredSlots"], 2)

    def test_insufficient_space_in_later_hdu_fails_whole_request(self):
        good = primary_hdu(checksums=False)  # HDU 0: plenty of space
        middle = image_hdu(b"\x01\x02", bitpix=8, axes=(2,),
                           checksums=False)
        # HDU 2: no room at all (5 structural + 30 comments + END fill
        # the 36 cards of one 2880-byte block exactly).
        tight = header_block(
            [card("XTENSION", "IMAGE"), card("BITPIX", 8), card("NAXIS", 0),
             card("PCOUNT", 0), card("GCOUNT", 1)]
            + [card("COMMENT", None) for _ in range(30)])
        with self.assertRaises(MaterializationFailure) as ctx:
            materialize_checksums(good + middle + tight)
        report = ctx.exception.report
        self.assertEqual(report["conclusion"], REJECTED)
        failure = report["failure"]
        self.assertEqual(failure["hdu"], 2)
        self.assertEqual(failure["reason"],
                         "INSUFFICIENT_HEADER_SPACE")
        # HDU 2 starts at 2880 + 2*2880; END is 35 cards in.
        self.assertEqual(failure["offset"], 2880 + 2 * 2880 + 35 * 80)

    def test_insufficient_space_report_is_audit_shaped(self):
        early = self._tight_primary(31)
        with self.assertRaises(MaterializationFailure) as ctx:
            materialize_checksums(early)
        report = ctx.exception.report
        self.assertEqual(set(report),
                         {"conclusion", "fileSize", "hduCount", "limits",
                          "hdus", "failure"})
        self.assertEqual(report["hduCount"], 1)


class AuditRejectionTests(unittest.TestCase):
    def _assert_blocked(self, blob, reason, hdu=None):
        with self.assertRaises(MaterializationFailure) as ctx:
            materialize_checksums(blob)
        report = ctx.exception.report
        self.assertEqual(report["conclusion"], REJECTED)
        self.assertEqual(report["failure"]["reason"], reason)
        if hdu is not None:
            self.assertEqual(report["failure"]["hdu"], hdu)
        # The report is exactly what the audit endpoint would return.
        self.assertEqual(report, audit_bytes(blob))

    def test_truncated_file_blocked(self):
        self._assert_blocked(fixtures.build_valid_file(1)[:6000],
                             "TRUNCATED_DATA", hdu=1)

    def test_trailing_bytes_blocked(self):
        self._assert_blocked(primary_hdu() + b"\x00" * 100,
                             "TRAILING_BYTES")

    def test_corrupt_datasum_blocked(self):
        blob = fixtures.corrupt_datasum(fixtures.build_valid_file(1),
                                        occurrence=1)
        self._assert_blocked(blob, "DATASUM_MISMATCH", hdu=1)

    def test_corrupt_checksum_blocked(self):
        blob = fixtures.corrupt_checksum(fixtures.build_valid_file(1),
                                         occurrence=1)
        self._assert_blocked(blob, "CHECKSUM_MISMATCH", hdu=1)

    def test_empty_file_blocked(self):
        self._assert_blocked(b"", "EMPTY_FILE")

    def test_nonspace_padding_blocked(self):
        # A non-space byte in padding that is not itself a checksum card
        # must keep the ordinary audit rejection (no file produced).
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        early = bytearray(header_block(cards))
        early[500] = 0x00
        self._assert_blocked(bytes(early), "NONSPACE_HEADER_PADDING")

    def test_lookalike_keyword_in_padding_is_not_a_card(self):
        # Slot starts with the DATASUM keyword but is not a value card:
        # it stays ordinary (bad) padding rather than becoming a card.
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        early = bytearray(header_block(cards))
        early[480:560] = b"DATASUM  not a value card".ljust(80)
        self._assert_blocked(bytes(early), "NONSPACE_HEADER_PADDING")

    def test_well_formed_checksum_card_after_end_is_padding(self):
        # Standard termination: even a correctly shaped DATASUM card
        # placed after the END card is not a card at all; it is
        # non-space header padding and rejects like any other padding
        # byte.  This guarantees the audit verdict matches a standard
        # reader that stops at the first END card.
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        early = bytearray(header_block(cards))
        early[480:560] = card("DATASUM", 12345)
        self._assert_blocked(bytes(early), "NONSPACE_HEADER_PADDING")

    def test_checksum_card_after_end_never_validates(self):
        # A complete DATASUM/CHECKSUM pair written after END (the old,
        # non-interoperable materialization output) must audit the same
        # way a standard reader sees it: the cards are invisible padding,
        # and the non-space bytes reject the file outright -- regardless
        # of whether the embedded values would otherwise be correct.
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        early = bytearray(header_block(cards))
        early[480:560] = card("DATASUM", sum32_be(b""))
        early[560:640] = card("CHECKSUM", "0" * 16)
        self._assert_blocked(bytes(early), "NONSPACE_HEADER_PADDING")


class StandardReaderInteropTests(unittest.TestCase):
    """Cross-checks with an independent reader that stops at first END."""

    def test_astropy_reference_is_recognized(self):
        with open(os.path.join(DATA_DIR, "valid_astropy.fits"), "rb") as fh:
            blob = fh.read()
        _assert_std_reader_accepts(self, blob, hdu_count=2)

    def test_both_missing_are_visible_before_end(self):
        early = fixtures.build_valid_file(2, checksums=False)
        completed = materialize_checksums(early)
        hdus = _assert_std_reader_accepts(self, completed, hdu_count=3)
        # The cards must literally occupy the slots ending at END.
        for hdu in hdus:
            end_off = hdu["end_offset"]
            self.assertEqual(
                completed[end_off - 160:end_off - 152], b"DATASUM ")
            self.assertEqual(
                completed[end_off - 80:end_off - 72], b"CHECKSUM")

    def test_only_checksum_missing(self):
        data = struct.pack(">6h", *range(6))
        datasum = sum32_be(data)
        ext_cards = [card("XTENSION", "IMAGE"), card("BITPIX", 16),
                     card("NAXIS", 1), card("NAXIS1", 6), card("PCOUNT", 0),
                     card("GCOUNT", 1), card("DATASUM", datasum)]
        early = primary_hdu(checksums=True) + header_block(ext_cards) \
            + data_block(data)
        completed = materialize_checksums(early)
        _assert_std_reader_accepts(self, completed, hdu_count=2)

    def test_only_datasum_missing(self):
        data = struct.pack(">6h", *range(6))
        datasum = sum32_be(data)
        ext_cards = [card("XTENSION", "IMAGE"), card("BITPIX", 16),
                     card("NAXIS", 1), card("NAXIS1", 6), card("PCOUNT", 0),
                     card("GCOUNT", 1), card("CHECKSUM", "0" * 16)]
        total = sum32_be(header_block(ext_cards), datasum)
        ext_cards[-1] = card("CHECKSUM", char_encode(~total & UINT32_MAX))
        early = primary_hdu(checksums=True) + header_block(ext_cards) \
            + data_block(data)
        completed = materialize_checksums(early)
        _assert_std_reader_accepts(self, completed, hdu_count=2)

    def test_exactly_enough_slots(self):
        # 33 content cards + END = 34 cards -> exactly 2 blank slots.
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0)]
        cards += [card("COMMENT", None) for _ in range(30)]
        early = header_block(cards)
        completed = materialize_checksums(early)
        _assert_std_reader_accepts(self, completed, hdu_count=1)
        # END lands on the final slot of the block.
        hdus = stdfits.read_hdus(completed)
        self.assertEqual(hdus[0]["end_offset"], 35 * 80)
        self.assertEqual(len(completed), 2880)

    def test_exactly_one_slot_for_one_missing_card(self):
        # One card already present: 34 content cards + END = 35 cards
        # leave exactly one blank slot, exactly enough for CHECKSUM.
        cards = [card("SIMPLE", True), card("BITPIX", 8), card("NAXIS", 0),
                 card("DATASUM", 0)]
        cards += [card("COMMENT", None) for _ in range(30)]
        early = header_block(cards)
        completed = materialize_checksums(early)
        _assert_std_reader_accepts(self, completed, hdu_count=1)
        hdus = stdfits.read_hdus(completed)
        self.assertEqual(hdus[0]["end_offset"], 35 * 80)

    def test_insufficient_space_fails_atomically_in_any_hdu(self):
        # HDU 0 and HDU 1 have room; HDU 2 has none.  No part of the
        # request may be materialized.
        good = primary_hdu(checksums=False)
        middle = image_hdu(b"\x01\x02", bitpix=8, axes=(2,),
                           checksums=False)
        tight_cards = [card("XTENSION", "IMAGE"), card("BITPIX", 8),
                       card("NAXIS", 0), card("PCOUNT", 0), card("GCOUNT", 1)]
        tight_cards += [card("COMMENT", None) for _ in range(30)]
        tight = header_block(tight_cards)
        early = good + middle + tight
        with self.assertRaises(MaterializationFailure) as ctx:
            materialize_checksums(early)
        self.assertEqual(ctx.exception.report["failure"]["hdu"], 2)
        self.assertEqual(
            ctx.exception.report["failure"]["reason"],
            "INSUFFICIENT_HEADER_SPACE")
        # Atomicity: materialize_checksums must not return a modified
        # buffer at all; re-running the audit on the original input shows
        # every HDU still missing its cards.
        self.assertEqual(
            [(h["datasum"]["verdict"], h["checksum"]["verdict"])
             for h in audit_bytes(early)["hdus"]],
            [("ABSENT", "ABSENT")] * 3)

    def test_multi_hdu_file_all_cards_before_end(self):
        early = fixtures.build_valid_file(3, checksums=False)
        completed = materialize_checksums(early)
        hdus = _assert_std_reader_accepts(self, completed, hdu_count=4)
        # Audit verdict and standard-reader verdict agree, HDU by HDU.
        report = audit_bytes(completed)
        self.assertEqual(report["conclusion"], ACCEPTED)
        for report_hdu, std_hdu in zip(report["hdus"], hdus):
            self.assertEqual(report_hdu["range"]["start"],
                             std_hdu["header_offset"])
            self.assertEqual(report_hdu["range"]["end"], std_hdu["padded_end"])
            self.assertEqual(report_hdu["datasum"]["verdict"], "VALID")
            self.assertEqual(report_hdu["checksum"]["verdict"], "VALID")

    def test_layout_is_preserved(self):
        early = fixtures.build_valid_file(2, checksums=False)
        completed = materialize_checksums(early)
        self.assertEqual(len(completed), len(early))
        before = audit_bytes(early)
        after = audit_bytes(completed)
        for b, a in zip(before["hdus"], after["hdus"]):
            self.assertEqual(b["range"], a["range"])
            self.assertEqual(b["header"]["offset"], a["header"]["offset"])
            self.assertEqual(b["header"]["bytes"], a["header"]["bytes"])
            self.assertEqual(b["data"], a["data"])
        # Standard reader sees the identical layout.
        std = stdfits.read_hdus(completed)
        for hdu, report_hdu in zip(std, after["hdus"]):
            self.assertEqual(hdu["header_bytes"],
                             report_hdu["header"]["bytes"])
            self.assertEqual(hdu["data_offset"],
                             report_hdu["data"]["offset"])
            self.assertEqual(hdu["padded_end"], report_hdu["range"]["end"])

    def test_science_payload_untouched_multi_hdu(self):
        early = fixtures.build_valid_file(3, checksums=False)
        completed = materialize_checksums(early)
        for hdu in audit_bytes(early)["hdus"]:
            start = hdu["data"]["offset"]
            end = start + hdu["data"]["paddedBytes"]
            self.assertEqual(completed[start:end], early[start:end])

    def test_complete_file_byte_for_byte_identical(self):
        with open(os.path.join(DATA_DIR, "valid_astropy.fits"), "rb") as fh:
            blob = fh.read()
        self.assertEqual(materialize_checksums(blob), blob)
        self.assertEqual(materialize_checksums(
            fixtures.build_max_hdus_file()), fixtures.build_max_hdus_file())

    def test_fixed_point(self):
        early = fixtures.build_valid_file(2, checksums=False)
        once = materialize_checksums(early)
        twice = materialize_checksums(once)
        self.assertEqual(once, twice)
        _assert_std_reader_accepts(self, once)


if __name__ == "__main__":
    unittest.main()
