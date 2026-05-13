"""
Run the LALAS pipeline on every xyz file in a directory or .tar.gz archive
and emit a CSV with columns:
    name           - xyz filename without the .xyz extension
    representation - full pipeline output "<ring types> <augmented LALAS>"
    lalas          - just the augmented LALAS part (the token after the
                     last space in ``representation``)

After generation, optionally compare the ``lalas`` column against
``data/all_data.csv``'s ``augmented_lalas`` column (joined on molecule
name) and report discrepancies (count + per-molecule listing).

Default input is the COMPAS-1D b3lyp tarball downloaded by the user;
``--input`` can also point at an already-extracted directory.

Examples (run from ``lala-repr-code/`` with the venv active):

    # full pipeline: generate reps from the tarball and compare
    python scripts/xyz_dir_to_lalas.py

    # custom input dir/archive
    python scripts/xyz_dir_to_lalas.py --input PATH/TO/dir-or-tarball

    # only compare a previously generated CSV
    python scripts/xyz_dir_to_lalas.py --skip-generate
"""

from __future__ import annotations

import argparse
import csv
import sys
import tarfile
import time
from io import TextIOWrapper
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LALA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = LALA_DIR.parent

sys.path.insert(0, str(LALA_DIR))

from xyz2lalas import (  # noqa: E402
    knotidentifier, pathfinder, chirality, ringclassifier, mol, molloader,
)


DEFAULT_INPUT = Path("C:/Users/ofree/Downloads/compas-1D_b3lyp_def2-svp.tar.gz")
DEFAULT_ALL_DATA = PROJECT_ROOT / "data" / "all_data.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results"

COVALENCY_FACTOR = 1.3


# ---------------------------------------------------------------- xyz reading

def _parse_xyz_lines(lines):
    """
    Parse an xyz file given as a list of lines. The first 2 lines are
    header (atom count, comment); the rest are atom lines
    ``ELEMENT  X  Y  Z``. Returns ``[[element, x, y, z], ...]``.
    """
    rows = []
    for i, line in enumerate(lines):
        if i < 2:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        if not symbol.isalpha():
            symbol = molloader.str_atom(int(symbol))
        rows.append([symbol.capitalize(), float(parts[1]), float(parts[2]), float(parts[3])])
    return rows


def iter_xyz_sources(input_path: Path):
    """
    Yield ``(name, lines)`` for every xyz under ``input_path``. ``input_path``
    can be a directory (recursively scanned for ``*.xyz``) or a ``.tar.gz`` /
    ``.tgz`` archive that contains xyz files (read in stream).
    """
    if input_path.is_dir():
        for xyz in sorted(input_path.rglob("*.xyz")):
            with open(xyz, "r", encoding="utf-8") as f:
                yield xyz.stem, f.readlines()
        return

    if input_path.is_file() and (
        input_path.suffixes[-2:] == [".tar", ".gz"]
        or input_path.suffix.lower() in {".tgz", ".gz"}
    ):
        with tarfile.open(input_path, "r:gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.lower().endswith(".xyz"):
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                text = TextIOWrapper(fobj, encoding="utf-8")
                yield Path(member.name).stem, text.readlines()
        return

    raise SystemExit(
        f"unsupported input (need a directory of .xyz files or a .tar.gz archive): {input_path}"
    )


# ----------------------------------------------------------------- pipeline

def lalas_representation(atom_rows):
    """
    Run the full LALAS pipeline on a list of ``[element, x, y, z]`` rows
    and return ``(representation, augmented_lalas)``.
    """
    molrepr = mol.Mol(atom_rows)
    molrepr.align_to_xy_plane()
    knots, _ = knotidentifier.identify(molrepr, COVALENCY_FACTOR)
    if len(knots) < 3:
        raise ValueError(f"only {len(knots)} ring(s); need at least 3")
    paths, graph, knotedges = pathfinder.find(knots)
    names_by_index = {
        k.index: ringclassifier.classify_ring(k, molrepr.atoms, COVALENCY_FACTOR)
        for k in knots
    }
    ring_types = ringclassifier.canonical_ring_sequence(
        paths, knotedges, knots, graph, molrepr.atoms, COVALENCY_FACTOR
    )
    _, augmented_lalas, _, _ = chirality.canonical_lalas_fields(
        paths, knotedges, knots, graph, ring_type_for_idx=names_by_index.get
    )
    return f"{ring_types} {augmented_lalas}", augmented_lalas


def generate(input_path: Path, out_csv: Path) -> int:
    """Generate the result CSV. Returns the number of successful rows."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    start = time.time()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "representation", "lalas"])
        for name, lines in iter_xyz_sources(input_path):
            try:
                rows = _parse_xyz_lines(lines)
                rep, lalas = lalas_representation(rows)
            except Exception as exc:
                n_fail += 1
                print(f"[fail] {name}: {exc}", flush=True)
                continue
            writer.writerow([name, rep, lalas])
            n_ok += 1
            if n_ok % 500 == 0:
                elapsed = time.time() - start
                rate = n_ok / elapsed if elapsed else 0.0
                print(f"... {n_ok} reps written  "
                      f"(failed: {n_fail}, elapsed: {elapsed:.1f}s, rate: {rate:.1f}/s)",
                      flush=True)
    elapsed = time.time() - start
    print(f"# generate: {n_ok} reps, {n_fail} failed, {elapsed:.1f}s -> {out_csv}", flush=True)
    return n_ok


# ----------------------------------------------------------------- compare

def compare_against_all_data(reps_csv: Path, all_data_csv: Path):
    """
    Stream both CSVs, join on molecule name, count matches/mismatches
    between the new ``lalas`` column and the reference ``augmented_lalas``
    column. Prints a summary and lists every mismatching molecule.
    """
    new_reps: dict[str, str] = {}
    with open(reps_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            new_reps[row["name"]] = row["lalas"]

    ref: dict[str, str] = {}
    with open(all_data_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "molecule" not in (r.fieldnames or []) or "augmented_lalas" not in (r.fieldnames or []):
            raise SystemExit(
                f"missing 'molecule' or 'augmented_lalas' column in {all_data_csv}"
            )
        for row in r:
            ref[row["molecule"]] = row["augmented_lalas"]

    only_in_new = sorted(set(new_reps) - set(ref))
    only_in_ref = sorted(set(ref) - set(new_reps))
    both = set(new_reps) & set(ref)

    matches = 0
    mismatches: list[tuple[str, str, str]] = []
    for name in sorted(both):
        if new_reps[name] == ref[name]:
            matches += 1
        else:
            mismatches.append((name, ref[name], new_reps[name]))

    print(f"\n# comparison: new={len(new_reps)}  reference={len(ref)}  common={len(both)}")
    print(f"  matches                        : {matches}")
    print(f"  mismatches                     : {len(mismatches)}")
    print(f"  only in new (no reference row) : {len(only_in_new)}")
    print(f"  only in reference (no new rep) : {len(only_in_ref)}")

    if mismatches:
        print("\n# mismatching molecules  (name | reference augmented_lalas | new lalas):")
        for name, ref_val, new_val in mismatches:
            print(f"  {name:40s} {ref_val!r:30s} -> {new_val!r}")

    if only_in_ref:
        print("\n# molecules present in reference but missing from new reps:")
        for name in only_in_ref[:50]:
            print(f"  {name}")
        if len(only_in_ref) > 50:
            print(f"  ... ({len(only_in_ref) - 50} more not shown)")


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help=f"Directory of .xyz files OR a .tar.gz archive. "
                             f"Default: {DEFAULT_INPUT}")
    parser.add_argument("--all-data-csv", default=str(DEFAULT_ALL_DATA),
                        help=f"Reference CSV for comparison. Default: {DEFAULT_ALL_DATA}")
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR),
                        help=f"Output directory. Default: {DEFAULT_RESULTS_DIR}")
    parser.add_argument("--out-name", default="compas-1D_b3lyp_lalas.csv",
                        help="Result CSV filename (placed in --out-dir).")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip generation; only run the comparison.")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Generate only; skip the comparison.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_csv = out_dir / args.out_name
    all_data_csv = Path(args.all_data_csv).expanduser()

    if not args.skip_generate:
        if not input_path.exists():
            raise SystemExit(f"input not found: {input_path}")
        generate(input_path, out_csv)

    if not args.skip_compare:
        if not out_csv.is_file():
            raise SystemExit(
                f"result CSV not found: {out_csv} (re-run without --skip-generate to create it)"
            )
        if not all_data_csv.is_file():
            raise SystemExit(f"all_data CSV not found: {all_data_csv}")
        compare_against_all_data(out_csv, all_data_csv)


if __name__ == "__main__":
    main()
