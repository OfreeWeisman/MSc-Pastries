"""
Extract a single named record (charge=0) from a large SDF into its own file.
Uses a seek-based scan so it doesn't have to read the entire file from the start.

Usage:
    python extract_molecule.py [NAME] [CHARGE]
    python extract_molecule.py C2M409023        # charge defaults to 0
    python extract_molecule.py C2M409023 -1
"""
import sys
import re
from pathlib import Path

NAME   = sys.argv[1] if len(sys.argv) > 1 else "C2M409023"
CHARGE = sys.argv[2] if len(sys.argv) > 2 else "0"

SDF_IN  = Path("examples/compas-2x.sdf")
SDF_OUT = Path(f"examples/{NAME}_chrg{CHARGE}.sdf")

TERMINATOR  = "$$$$"
DATA_HEADER = re.compile(r"^>.*<\s*([^>]+?)\s*>")
TARGET_UPPER = NAME.upper()


def _read_data_fields(lines):
    """Parse > <KEY> / value blocks from record lines (skipping atom/bond block)."""
    data = {}
    key  = None
    vals = []
    try:
        n_atoms = int(lines[3][0:3])
        n_bonds = int(lines[3][3:6])
    except (IndexError, ValueError):
        return data
    start = 4 + n_atoms + n_bonds
    for ln in lines[start:]:
        s = ln.rstrip("\r\n")
        if s.strip() == TERMINATOR:
            if key:
                data[key] = "\n".join(vals).strip()
            break
        m = DATA_HEADER.match(s)
        if m:
            if key:
                data[key] = "\n".join(vals).strip()
            key, vals = m.group(1).strip(), []
        elif key is not None:
            vals.append(s)
    return data


def _seek_to_record_boundary(f, byte_offset):
    """Seek to byte_offset, discard the partial line, then advance past the
    next $$$$ so the next readline() returns a record title line."""
    f.seek(byte_offset)
    f.readline()
    for line in f:
        if line.rstrip("\r\n").strip() == TERMINATOR:
            return


def _collect_record(first_line, f):
    """Given the title line, read until $$$$ and return all lines."""
    buf = [first_line]
    for line in f:
        buf.append(line)
        if line.rstrip("\r\n").strip() == TERMINATOR:
            break
    return buf


# Two-pass: start at ~60 % of the file first (C2Mxxxxxx records tend to be
# in the second half), then fall back to scanning from byte 0.
file_size    = SDF_IN.stat().st_size
seek_offsets = [int(file_size * 0.60), 0]
found_lines  = None

with open(SDF_IN, "r", encoding="utf-8", errors="replace") as f:
    for start_offset in seek_offsets:
        _seek_to_record_boundary(f, start_offset)
        n_checked = 0

        for line in f:
            title = line.rstrip("\r\n").strip()
            if title.upper() != TARGET_UPPER:
                # Skip to the next record boundary.
                for skip in f:
                    if skip.rstrip("\r\n").strip() == TERMINATOR:
                        break
                n_checked += 1
                if n_checked % 50000 == 0:
                    print(f"  scanned {n_checked} records from offset "
                          f"{start_offset/1e9:.2f} GB …")
                continue

            # Title matches – collect the full record.
            buf    = _collect_record(line, f)
            fields = _read_data_fields(buf)
            if fields.get("charge", "").strip() == CHARGE:
                found_lines = buf
                break
            # Wrong charge; keep scanning.

        if found_lines:
            break
        if start_offset > 0:
            print(f"  not found from {start_offset/1e9:.2f} GB; "
                  f"retrying from start …")

if not found_lines:
    sys.exit(f"ERROR: '{NAME}' charge={CHARGE} not found in {SDF_IN}")

SDF_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(SDF_OUT, "w", encoding="utf-8") as f:
    f.writelines(found_lines)
print(f"Wrote {len(found_lines)} lines -> {SDF_OUT}")
