"""One-shot verification service.

Runs three verification stages and summarizes them in the process exit
code (bit flags, so failures compose):

* bit 0 (1): code tests      -- the unit test suite (unittest discover)
* bit 1 (2): application build -- byte-compilation and import of the app
* bit 2 (4): HTTP smoke      -- audit verdicts over the live API for a
  valid file, digest-corrupted files and a truncated file

Exit code 0 means every stage passed.  The service is meant to be run
once (``docker compose up --exit-code-from verify verify``) and then
exit by itself.
"""

import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = os.environ.get("API_URL", "http://api:8000").rstrip("/")
HEALTH_TIMEOUT = float(os.environ.get("VERIFY_HEALTH_TIMEOUT", "60"))

EXIT_TESTS = 1
EXIT_BUILD = 2
EXIT_SMOKE = 4


def _banner(title):
    print("\n=== %s %s" % (title, "=" * (60 - len(title))), flush=True)


def run_tests():
    _banner("stage 1/3: code tests (python -m unittest)")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests",
         "-t", ".", "-v"],
        cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    ok = proc.returncode == 0
    print("code tests: %s" % ("PASS" if ok else "FAIL"), flush=True)
    return ok


def run_build():
    _banner("stage 2/3: application build (byte-compile + import)")
    compile_proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "fitsaudit", "verify",
         "tests"],
        cwd=ROOT, capture_output=True, text=True)
    if compile_proc.returncode != 0:
        sys.stderr.write(compile_proc.stderr)
        print("application build: FAIL (byte-compilation)", flush=True)
        return False
    import_proc = subprocess.run(
        [sys.executable, "-c",
         "import fitsaudit.server, fitsaudit.core, fitsaudit.fixtures; "
         "print('imports ok')"],
        cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(import_proc.stdout)
    sys.stderr.write(import_proc.stderr)
    ok = import_proc.returncode == 0
    print("application build: %s" % ("PASS" if ok else "FAIL"), flush=True)
    return ok


def _post(path, blob, content_type="application/fits", raw=False):
    req = urllib.request.Request(
        API_URL + path, data=blob, method="POST",
        headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if raw:
                return resp.status, ctype, body
            return resp.status, _decode_json(body)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        ctype = exc.headers.get("Content-Type", "") if exc.headers else ""
        if raw:
            return exc.code, ctype, body
        return exc.code, _decode_json(body)


def _decode_json(body):
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError:
        return {}


def _get(path):
    with urllib.request.urlopen(API_URL + path, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _wait_for_health():
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        try:
            status, payload = _get("/health")
            if status == 200 and payload.get("status") == "ok":
                print("API healthy at %s" % API_URL, flush=True)
                return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.5)
    print("API at %s did not become healthy within %.0fs"
          % (API_URL, HEALTH_TIMEOUT), flush=True)
    return False


def run_smoke():
    _banner("stage 3/3: HTTP smoke verdicts against %s" % API_URL)
    if not _wait_for_health():
        return False

    sys.path.insert(0, ROOT)
    from fitsaudit import fixtures  # noqa: PLC0415
    from fitsaudit.core import (  # noqa: PLC0415
        audit_bytes,
        char_encode,
        sum32_be,
        UINT32_MAX,
    )
    from verify import stdfits  # noqa: PLC0415

    valid = open(os.path.join(ROOT, "tests", "data",
                              "valid_astropy.fits"), "rb").read()
    built = fixtures.build_valid_file(2)
    corrupt_datasum = fixtures.corrupt_datasum(valid, occurrence=1)
    corrupt_checksum = fixtures.corrupt_checksum(valid, occurrence=1)
    truncated = valid[:6000]  # cut inside HDU 1's data section
    trailing = valid + b"\x00" * 100

    checks = []

    def record(name, ok, detail=""):
        ok = bool(ok)
        checks.append(ok)
        print("%-44s %s%s" % (
            name, "PASS" if ok else "FAIL",
            "" if ok else " (%s)" % detail), flush=True)
        return ok

    def expect(name, blob, want_status, want_conclusion=None,
               want_reason=None, want_hdu=None, want_hdus=None):
        status, report = _post("/api/fits/audit", blob)
        problems = []
        if status != want_status:
            problems.append("status %d != %d" % (status, want_status))
        if want_conclusion is not None \
                and report.get("conclusion") != want_conclusion:
            problems.append("conclusion %r != %r"
                            % (report.get("conclusion"), want_conclusion))
        failure = report.get("failure") or {}
        if want_reason is not None \
                and failure.get("reason") != want_reason:
            problems.append("reason %r != %r"
                            % (failure.get("reason"), want_reason))
        if want_hdu is not None and failure.get("hdu") != want_hdu:
            problems.append("hdu %r != %r" % (failure.get("hdu"), want_hdu))
        if want_hdus is not None and report.get("hduCount") != want_hdus:
            problems.append("hduCount %r != %r"
                            % (report.get("hduCount"), want_hdus))
        record(name, not problems, "; ".join(problems))
        return report

    materialize_path = "/api/fits/checksums/materialize"

    def materialize(blob):
        return _post(materialize_path, blob, raw=True)

    # -- audit endpoint regression coverage ------------------------------

    # A valid file (generated by astropy) must be accepted.
    report = expect("valid file accepted", valid, 200,
                    want_conclusion="ACCEPTED", want_hdus=2)
    if report.get("hdus"):
        verdicts = [(h["datasum"]["verdict"], h["checksum"]["verdict"])
                    for h in report["hdus"]]
        record("valid file checksum verdicts",
               all(v == ("VALID", "VALID") for v in verdicts),
               "%r" % verdicts)

    # A valid file built independently (integer DATASUM form, 3 HDUs).
    expect("built valid file accepted", built, 200,
           want_conclusion="ACCEPTED", want_hdus=3)

    # Corrupted digests must be rejected with the matching reason code,
    # the earliest failing HDU and a byte offset.
    expect("corrupt DATASUM rejected", corrupt_datasum, 200,
           want_conclusion="REJECTED", want_reason="DATASUM_MISMATCH",
           want_hdu=1)
    expect("corrupt CHECKSUM rejected", corrupt_checksum, 200,
           want_conclusion="REJECTED", want_reason="CHECKSUM_MISMATCH",
           want_hdu=1)

    # A truncated file must be rejected.
    expect("truncated file rejected", truncated, 200,
           want_conclusion="REJECTED", want_reason="TRUNCATED_DATA",
           want_hdu=1)

    # Trailing bytes must be rejected as well.
    expect("trailing bytes rejected", trailing, 200,
           want_conclusion="REJECTED", want_reason="TRAILING_BYTES")

    # Wrong media type is a request-level error.
    status, _ = _post("/api/fits/audit", valid,
                      content_type="application/octet-stream")
    record("wrong media type -> 415", status == 415, "status %d" % status)

    # -- independent standard-reader cross-check helpers -----------------

    def assert_std_reader_ok(name, blob, want_hdus):
        """An independent reader (stops at first END) must accept the
        materialized file, and the audit endpoint must reach the same
        HDU boundaries and VALID/VALID verdicts for every HDU."""
        try:
            ok_std, std_hdus = stdfits.verify_materialized(blob)
        except stdfits.StdFitsError as exc:
            record(name, False,
                   "standard reader rejected: %s at offset %d"
                   % (exc.reason, exc.offset))
            return None
        status, re_report = _post("/api/fits/audit", blob)
        audit_hdus = re_report.get("hdus", [])
        problems = []
        if not ok_std:
            problems.append("standard reader did not verify every card")
        if status != 200 or re_report.get("conclusion") != "ACCEPTED":
            problems.append("audit status=%d conclusion=%r"
                            % (status, re_report.get("conclusion")))
        if len(audit_hdus) != want_hdus or len(std_hdus) != want_hdus:
            problems.append("hdu count audit=%d std=%d want=%d"
                            % (len(audit_hdus), len(std_hdus), want_hdus))
        for i, (a, s) in enumerate(zip(audit_hdus, std_hdus)):
            if a["range"]["start"] != s["header_offset"] \
                    or a["range"]["end"] != s["padded_end"]:
                problems.append("HDU %d boundary disagreement" % i)
            if a["header"]["bytes"] != s["header_bytes"]:
                problems.append("HDU %d header-size disagreement" % i)
            if (a["datasum"].get("verdict") != "VALID"
                    or a["checksum"].get("verdict") != "VALID"):
                problems.append("HDU %d audit verdict not VALID/VALID" % i)
            for which in ("datasum", "checksum"):
                v = s[which]
                if not (v["present"] and v["valid"]
                        and v["card_before_end"]):
                    problems.append("HDU %d %s not a valid pre-END card: %r"
                                    % (i, which, v))
        record(name, not problems, "; ".join(problems))
        return std_hdus if not problems else None

    def assert_layout_payload(name, early, completed):
        """Header block size, data offsets, HDU boundaries and the
        scientific payload are all unchanged byte for byte."""
        before = audit_bytes(early)
        after = audit_bytes(completed)
        problems = []
        if len(completed) != len(early):
            problems.append("file length changed")
        for b, a in zip(before["hdus"], after["hdus"]):
            if b["range"] != a["range"]:
                problems.append("HDU %d range changed" % b["index"])
            if b["header"]["bytes"] != a["header"]["bytes"]:
                problems.append("HDU %d header block size changed"
                                % b["index"])
            if b["data"] != a["data"]:
                problems.append("HDU %d data section moved/resized"
                                % b["index"])
            start = b["data"]["offset"]
            end = start + b["data"]["paddedBytes"]
            if completed[start:end] != early[start:end]:
                problems.append("HDU %d science payload changed"
                                % b["index"])
        record(name, not problems, "; ".join(problems))

    # -- materialization: case 1, BOTH cards missing (multi-HDU) --------

    early = fixtures.build_valid_file(2, checksums=False)
    status, ctype, completed = materialize(early)
    framed = (status == 200 and ctype == "application/fits"
              and completed != early and len(completed) == len(early))
    record("both missing: 200 application/fits, same length", framed,
           "status %d, %r" % (status, ctype))
    if framed:
        std_hdus = assert_std_reader_ok(
            "both missing: standard reader verifies before END",
            completed, 3)
        if std_hdus is not None:
            order_ok = True
            for hdu in std_hdus:
                e = hdu["end_offset"]
                if not (completed[e - 160:e - 152] == b"DATASUM "
                        and completed[e - 80:e - 72] == b"CHECKSUM"
                        and completed[e:e + 80] == fixtures.END_CARD
                        and completed[e + 80:hdu["header_offset"]
                                      + hdu["header_bytes"]].strip(b" ")
                        == b""):
                    order_ok = False
            record("both missing: order DATASUM, CHECKSUM, END, spaces",
                   order_ok)
        assert_layout_payload(
            "both missing: header blocks, data, boundaries, payload",
            early, completed)

    # -- case 2: complete input passes through byte for byte -----------

    for label, blob in (("astropy reference", valid),
                        ("built complete file", built),
                        ("max-HDU file", fixtures.build_max_hdus_file())):
        status, ctype, passthrough = materialize(blob)
        record("complete input passthrough: " + label,
               status == 200 and ctype == "application/fits"
               and passthrough == blob, "status %d" % status)

    # -- case 3: idempotent retry --------------------------------------

    status, ctype, retried = materialize(completed)
    record("idempotent retry byte-identical",
           status == 200 and ctype == "application/fits"
           and retried == completed, "status %d" % status)

    # -- helper: an extension HDU carrying exactly one checksum card ---

    def extension_with_one(which):
        payload = struct.pack(">6h", *range(6))
        datasum = sum32_be(payload)
        ecards = [fixtures.card("XTENSION", "IMAGE"),
                  fixtures.card("BITPIX", 16),
                  fixtures.card("NAXIS", 1),
                  fixtures.card("NAXIS1", 6),
                  fixtures.card("PCOUNT", 0),
                  fixtures.card("GCOUNT", 1)]
        if which == "datasum":
            ecards.append(fixtures.card("DATASUM", datasum))
        else:
            ecards.append(fixtures.card("CHECKSUM", "0" * 16))
            total = sum32_be(fixtures.header_block(ecards), datasum)
            ecards[-1] = fixtures.card(
                "CHECKSUM", char_encode(~total & UINT32_MAX))
        return (fixtures.primary_hdu(checksums=True)
                + fixtures.header_block(ecards)
                + fixtures.data_block(payload))

    # -- case 4: only CHECKSUM missing ---------------------------------

    one_datasum = extension_with_one("datasum")
    status, ctype, completed = materialize(one_datasum)
    record("only checksum missing: 200 fits",
           status == 200 and ctype == "application/fits",
           "status %d" % status)
    if status == 200:
        std_hdus = assert_std_reader_ok(
            "only checksum missing: standard reader verifies",
            completed, 2)
        assert_layout_payload(
            "only checksum missing: layout and payload",
            one_datasum, completed)
        if std_hdus is not None:
            ds_off = fixtures.find_card(one_datasum, "DATASUM", 1)
            old_end = fixtures.find_card(one_datasum, "END", 1)
            record("only checksum missing: DATASUM card unmoved",
                   completed[ds_off:ds_off + 80]
                   == one_datasum[ds_off:ds_off + 80])
            new_end = old_end + 80
            record("only checksum missing: CHECKSUM directly before END",
                   completed[new_end - 80:new_end - 72] == b"CHECKSUM"
                   and completed[new_end:new_end + 80]
                   == fixtures.END_CARD)

    # -- case 5: only DATASUM missing (CHECKSUM recomputed in place) ---

    one_checksum = extension_with_one("checksum")
    old_checksum_off = fixtures.find_card(one_checksum, "CHECKSUM", 1)
    status, ctype, completed = materialize(one_checksum)
    record("only datasum missing: 200 fits",
           status == 200 and ctype == "application/fits",
           "status %d" % status)
    if status == 200:
        std_hdus = assert_std_reader_ok(
            "only datasum missing: standard reader verifies",
            completed, 2)
        assert_layout_payload(
            "only datasum missing: layout and payload",
            one_checksum, completed)
        if std_hdus is not None:
            record("only datasum missing: CHECKSUM card slot fixed",
                   fixtures.find_card(completed, "CHECKSUM", 1)
                   == old_checksum_off)
            old_end = fixtures.find_card(one_checksum, "END", 1)
            new_end = old_end + 80
            record("only datasum missing: DATASUM directly before END",
                   completed[new_end - 80:new_end - 72] == b"DATASUM "
                   and completed[new_end:new_end + 80]
                   == fixtures.END_CARD)

    # -- case 6: available slots exactly enough ------------------------

    exact_cards = [fixtures.card("SIMPLE", True), fixtures.card("BITPIX", 8),
                   fixtures.card("NAXIS", 0)]
    exact_cards += [fixtures.card("COMMENT", None) for _ in range(30)]
    exact = fixtures.header_block(exact_cards)  # 34 cards -> 2 free slots
    status, ctype, completed = materialize(exact)
    record("exactly enough slots: 200 fits",
           status == 200 and ctype == "application/fits",
           "status %d" % status)
    if status == 200:
        std_hdus = assert_std_reader_ok(
            "exactly enough slots: standard reader verifies", completed, 1)
        if std_hdus is not None:
            record("exactly enough slots: END on the last block slot",
                   std_hdus[0]["end_offset"] == 35 * 80
                   and len(completed) == 2880)

    # One missing card with exactly one free slot also just fits.
    tight_one_cards = [
        fixtures.card("SIMPLE", True), fixtures.card("BITPIX", 8),
        fixtures.card("NAXIS", 0), fixtures.card("DATASUM", 0)]
    tight_one_cards += [fixtures.card("COMMENT", None) for _ in range(30)]
    tight_one = fixtures.header_block(tight_one_cards)  # 35 cards -> 1 slot
    status, _, completed_one = materialize(tight_one)
    if status == 200:
        assert_std_reader_ok(
            "exactly one slot for one missing card", completed_one, 1)
    else:
        record("exactly one slot for one missing card", False,
               "status %d" % status)

    # -- case 7: a single HDU short by one slot -> atomic 422 ----------

    tight_cards = [fixtures.card("SIMPLE", True), fixtures.card("BITPIX", 8),
                   fixtures.card("NAXIS", 0)]
    tight_cards += [fixtures.card("COMMENT", None) for _ in range(31)]
    tight = fixtures.header_block(tight_cards)
    status, ctype, raw_body = materialize(tight)
    space_report = _decode_json(raw_body)
    failure = space_report.get("failure") or {}
    details = failure.get("details") or {}
    record("insufficient header space -> 422",
           status == 422 and ctype == "application/json"
           and space_report.get("conclusion") == "REJECTED"
           and failure.get("hdu") == 0
           and failure.get("reason") == "INSUFFICIENT_HEADER_SPACE"
           and failure.get("offset") == 34 * 80
           and details.get("endOffset") == 34 * 80
           and details.get("availableSlots") == 1
           and details.get("requiredSlots") == 2,
           "status %d, %r" % (status, failure))

    # -- case 8: one tight HDU in a multi-HDU file fails the whole ----
    # request atomically: JSON 422, never a partially materialized FITS
    # body, and re-auditing the (untouched) request bytes still shows
    # every HDU as ABSENT/ABSENT.
    good = fixtures.primary_hdu(checksums=False)
    middle = fixtures.image_hdu(b"\x01\x02", bitpix=8, axes=(2,),
                                checksums=False)
    later_tight_cards = [
        fixtures.card("XTENSION", "IMAGE"), fixtures.card("BITPIX", 8),
        fixtures.card("NAXIS", 0), fixtures.card("PCOUNT", 0),
        fixtures.card("GCOUNT", 1)]
    later_tight_cards += [fixtures.card("COMMENT", None) for _ in range(30)]
    later_tight = fixtures.header_block(later_tight_cards)
    multi = good + middle + later_tight
    status, ctype, raw_body = materialize(multi)
    later_report = _decode_json(raw_body)
    later_failure = later_report.get("failure") or {}
    problems = []
    if status != 422:
        problems.append("status %d != 422" % status)
    if ctype != "application/json":
        problems.append("content-type %r" % ctype)
    if later_failure.get("hdu") != 2 \
            or later_failure.get("reason") != "INSUFFICIENT_HEADER_SPACE":
        problems.append("failure %r" % later_failure)
    fresh_status, fresh = _post("/api/fits/audit", multi)
    verdicts = [(h["datasum"]["verdict"], h["checksum"]["verdict"])
                for h in fresh.get("hdus", [])]
    if fresh_status != 200 or verdicts != [("ABSENT", "ABSENT")] * 3:
        problems.append("post-failure verdicts: %r" % verdicts)
    record("tight HDU in multi-HDU file fails whole request atomically",
           not problems, "; ".join(problems))

    # -- case 9: structurally rejected input -> 422, no file ----------

    status, ctype, raw_body = materialize(valid[:6000])
    blocked = _decode_json(raw_body)
    record("truncated input blocked -> 422",
           status == 422 and ctype == "application/json"
           and (blocked.get("failure") or {}).get("reason")
           == "TRUNCATED_DATA", "status %d" % status)

    # -- case 10: request-level error ----------------------------------

    status, _, _ = _post(materialize_path, exact,
                         content_type="application/octet-stream", raw=True)
    record("materialize wrong media type -> 415", status == 415,
           "status %d" % status)

    ok = all(checks)
    print("HTTP smoke: %s (%d/%d checks passed)"
          % ("PASS" if ok else "FAIL", sum(checks), len(checks)),
          flush=True)
    return ok


def main():
    tests_ok = run_tests()
    build_ok = run_build()
    smoke_ok = run_smoke()

    exit_code = 0
    if not tests_ok:
        exit_code |= EXIT_TESTS
    if not build_ok:
        exit_code |= EXIT_BUILD
    if not smoke_ok:
        exit_code |= EXIT_SMOKE

    _banner("verify summary")
    summary = {
        "tests": "PASS" if tests_ok else "FAIL",
        "build": "PASS" if build_ok else "FAIL",
        "smoke": "PASS" if smoke_ok else "FAIL",
        "exitCode": exit_code,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
