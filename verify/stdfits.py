"""Minimal, strictly standard FITS reader used only by the verify chain.

This reader independently re-implements just enough of the FITS
standard to locate HDUs and recognize/verify ``DATASUM`` / ``CHECKSUM``
cards, so the materialized output can be cross-checked without reusing
the auditor's own parser:

* header cards are read one 80-byte record at a time and the header
  terminates **unconditionally at the first ``END`` card** -- slots
  after it are space padding and are never interpreted as cards (this is
  exactly the termination semantics of CFITSIO/astropy);
* data-section length follows ``BITPIX`` / ``NAXIS`` / ``NAXISn`` /
  ``PCOUNT`` / ``GCOUNT`` and is zero-padded to 2880 bytes;
* ``DATASUM`` is verified against the 1's-complement sum of the data
  words; ``CHECKSUM`` is verified by the convention's all-ones identity
  -- folding the stored header (CHECKSUM value in place) together with
  the data yields ``0xFFFFFFFF`` -- so no checksum encoder is shared
  with the auditor.

Nothing here is imported by the service itself; the one-shot ``verify``
service (and the unit tests) use it to prove the materialized cards are
visible to a reader that stops at END.
"""

from __future__ import annotations

CARD_SIZE = 80
BLOCK_SIZE = 2880
UINT32_MAX = 0xFFFFFFFF
BLANK_CARD = b" " * CARD_SIZE


class StdFitsError(Exception):
    """The file is not readable as a standard FITS file."""

    def __init__(self, reason, offset, message=""):
        super().__init__(message or "%s at offset %d" % (reason, offset))
        self.reason = reason
        self.offset = offset


def sum32(buf, seed=0):
    """Independent 1's-complement big-endian uint32 folding sum."""
    total = seed
    for off in range(0, len(buf), 4):
        word = int.from_bytes(buf[off:off + 4].ljust(4, b"\x00"), "big")
        total += word
        if total > UINT32_MAX:  # fold carries back into the low 32 bits
            total = (total & UINT32_MAX) + (total >> 32)
    return total


def _round_up(n, multiple):
    return n if n % multiple == 0 else n + multiple - n % multiple


def _integer_value(image):
    field = image[10:]
    if b"'" in field:  # astropy's quoted-string DATASUM form
        text = field.split(b"'", 2)[1].strip()
    else:
        text = field.split(b"/", 1)[0].strip()
    return int(text)


def _string_span(image):
    """Card-relative (start, end) indices of a quoted string value."""
    i = 10
    while image[i:i + 1] == b" ":
        i += 1
    if image[i:i + 1] != b"'":
        raise ValueError("not a string card")
    j = i + 1
    while True:
        if image[j:j + 1] == b"'":
            if image[j + 1:j + 2] == b"'":
                j += 2
                continue
            break
        j += 1
    return i + 1, j  # inside the quotes


def read_hdus(data):
    """Walk *data* with standard END-stopping semantics.

    Returns a list of dicts (one per HDU) describing the layout and the
    checksum cards seen *before* the END card.  Raises
    :class:`StdFitsError` on any structural problem.
    """
    hdus = []
    pos = 0
    index = 0
    while pos < len(data):
        if len(data) - pos < BLOCK_SIZE:
            raise StdFitsError(
                "TRAILING_BYTES", pos,
                "%d trailing byte(s) at offset %d" % (len(data) - pos, pos))
        hdr_start = pos
        cards = {}
        first_keyword = None
        while True:
            if pos + CARD_SIZE > len(data):
                raise StdFitsError("TRUNCATED_HEADER", hdr_start)
            image = data[pos:pos + CARD_SIZE]
            if min(image) < 0x20 or max(image) > 0x7E:
                raise StdFitsError("INVALID_CARD", pos)
            keyword = image[:8].rstrip(b" ").decode("ascii")
            if first_keyword is None:
                first_keyword = keyword
            if keyword == "END":
                end_card = pos
                pos += CARD_SIZE
                break
            cards[keyword] = pos  # last occurrence wins; audit forbids dupes
            pos += CARD_SIZE

        hdr_end = hdr_start + _round_up(pos - hdr_start, BLOCK_SIZE)
        if hdr_end > len(data):
            raise StdFitsError("TRUNCATED_HEADER", hdr_start)
        # Standard termination: every byte after the first END card is
        # padding and must be an ASCII space.
        header_padding = data[pos:hdr_end]
        if header_padding != b" " * len(header_padding):
            first = next(i for i, b in enumerate(header_padding)
                         if b != 0x20)
            raise StdFitsError("NONSPACE_HEADER_PADDING", pos + first)

        def need(keyword):
            if keyword not in cards:
                raise StdFitsError("MISSING_KEYWORD", end_card)
            return cards[keyword]

        if index == 0:
            if first_keyword != "SIMPLE":
                raise StdFitsError("MISSING_KEYWORD", hdr_start)
            hdu_type = "PRIMARY"
        else:
            if first_keyword != "XTENSION":
                raise StdFitsError("MISSING_KEYWORD", hdr_start)
            hdu_type = "IMAGE"

        bitpix = _integer_value(data[need("BITPIX"):need("BITPIX") + 80])
        naxis = _integer_value(data[need("NAXIS"):need("NAXIS") + 80])
        elements = 1
        for n in range(1, naxis + 1):
            off = need("NAXIS%d" % n)
            elements *= _integer_value(data[off:off + 80])
        if naxis == 0:
            elements = 0
        pcount = (_integer_value(data[cards["PCOUNT"]:cards["PCOUNT"] + 80])
                  if "PCOUNT" in cards else 0)
        gcount = (_integer_value(data[cards["GCOUNT"]:cards["GCOUNT"] + 80])
                  if "GCOUNT" in cards else 1)
        data_bytes = (abs(bitpix) // 8) * gcount * (pcount + elements)

        data_start = hdr_end
        data_end = data_start + data_bytes
        padded_end = data_start + _round_up(data_bytes, BLOCK_SIZE)
        if padded_end > len(data):
            raise StdFitsError("TRUNCATED_DATA", data_start)
        if data[data_end:padded_end].strip(b"\x00"):
            raise StdFitsError("NONZERO_DATA_PADDING", data_end)

        hdu = {
            "index": index,
            "type": hdu_type,
            "header_offset": hdr_start,
            "header_bytes": hdr_end - hdr_start,
            "end_offset": end_card,
            "data_offset": data_start,
            "data_bytes": data_bytes,
            "padded_end": padded_end,
            "datasum": {"present": False, "valid": False, "card_before_end": False},
            "checksum": {"present": False, "valid": False,
                         "card_before_end": False},
        }

        data_section = data[data_start:data_end]
        computed_datasum = sum32(data_section)
        if "DATASUM" in cards:
            off = cards["DATASUM"]
            stored = _integer_value(data[off:off + 80])
            hdu["datasum"].update(
                present=True, card_before_end=off < end_card,
                valid=(stored == computed_datasum),
                stored=stored, computed=computed_datasum)

        if "CHECKSUM" in cards:
            off = cards["CHECKSUM"]
            image = data[off:off + 80]
            try:
                q0, q1 = _string_span(image)
                sixteen = q1 - q0 == 16
            except ValueError:
                sixteen = False
            # Convention identity: the stored header (CHECKSUM value
            # left in place) folded with the data DATASUM is all ones.
            valid = (sixteen
                     and sum32(data[hdr_start:hdr_end],
                               computed_datasum) == UINT32_MAX)
            hdu["checksum"].update(
                present=True, card_before_end=(off < end_card and sixteen),
                valid=valid)

        hdus.append(hdu)
        pos = padded_end
        index += 1
    return hdus


def verify_materialized(data):
    """Strict standard-reader verdict for a materialized file.

    Returns ``(ok, hdus)`` where *ok* is true only when the file ends
    exactly at an HDU boundary and every HDU exposes a present, valid
    DATASUM and CHECKSUM card located before its terminating END card.
    """
    hdus = read_hdus(data)
    ok = bool(hdus)
    for hdu in hdus:
        for name in ("datasum", "checksum"):
            verdict = hdu[name]
            if not (verdict["present"] and verdict["valid"]
                    and verdict["card_before_end"]):
                ok = False
    return ok, hdus
