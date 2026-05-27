"""
Generate the LALAS (augmented) representation for every charge-0 molecule in a
COMPAS SDF (.sdf or .sdf.gz) and emit a CSV ``name,representation``.

Optionally, merge that CSV with the COMPAS feature CSV
(``data/compas-2x.csv``) - also restricted to charge-0 rows - to produce a
final combined CSV with columns
``name, formula, representation, <every other feature column in its original
order>``.

The SDF is iterated record-by-record (never loaded fully) so even the 1 GB
``compas-2x.sdf.gz`` is fine. Each molecule's representation is computed by
the same pipeline used by ``xyz2lalas`` (knot identification, path finding,
ring classification, augmented LALAS).

Typical usage (from the lala-repr-code/ directory, venv activated):

    # 1. Full run with defaults: generate reps + merge.
    python scripts/sdf2lalas_csv.py

    # 2. Quick test on the first 20 charge-0 molecules.
    python scripts/sdf2lalas_csv.py --limit 20 \
        --rep-name compas-2x_lalas_sample.csv \
        --final-name compas-2x_lalas_features_sample.csv

    # 3. Skip generation and just (re)merge an existing reps CSV.
    python scripts/sdf2lalas_csv.py --skip-reps

Outputs are written under ``data/results/`` at the project root.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LALA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = LALA_DIR.parent

# So we can import ``xyz2lalas`` as a package and ``compas2lalas`` as a
# top-level module sitting next to this script.
sys.path.insert(0, str(LALA_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from xyz2lalas import knotidentifier, pathfinder, chirality, ringclassifier, mol  # noqa: E402
from compas2lalas import iter_sdf  # noqa: E402


DEFAULT_SDF = Path("C:/Users/ofree/Desktop/Projects/compas/COMPAS-2/compas-2x.sdf.gz")
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "compas-2x.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# Same value used by xyz2lalas/main.py.
COVALENCY_FACTOR = 1.3

# csv module's field-size default is too small for the very long inchi / smiles
# strings in compas-2x.csv. Raise to a comfortable upper bound.
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


# --------------------------------------------------------------- pipeline core

def lalas_representation(atoms, bonds=None):
    """
    Run the full LALAS pipeline on a list of (element, x, y, z) tuples and
    return the augmented representation string:
        "<ring types> <augmented LALAS>"

    ``bonds``, when provided, is the explicit SDF bond list (0-based atom
    index pairs). When passed, the knot identifier uses those bonds
    directly instead of inferring connectivity from geometry, so the
    representation is fully determined by the input file's chemistry.
    """
    rows = [[el, x, y, z] for (el, x, y, z) in atoms]
    molrepr = mol.Mol(rows, bonds)
    molrepr.align_to_xy_plane()

    knots, _ = knotidentifier.identify(molrepr, COVALENCY_FACTOR)
    if len(knots) < 3:
        raise ValueError(f"only {len(knots)} ring(s) identified; need at least 3")

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
    return f"{ring_types} {augmented_lalas}"


# ----------------------------------------------------- step 1: generate reps

def generate_representations_csv(sdf_path: Path, out_csv: Path,
                                 charge_value: str = "0",
                                 limit: int | None = None,
                                 resume: bool = False) -> int:
    """
    Iterate the SDF, keep only records whose ``charge`` data field matches
    ``charge_value``, compute the LALAS representation, and stream the result
    to ``out_csv`` (columns: name, representation).

    Returns the number of successful rows written.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_names: set[str] = set()
    write_header = True
    mode = "w"
    if resume and out_csv.is_file():
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "name" not in reader.fieldnames:
                raise SystemExit(f"cannot resume; reps CSV missing name column: {out_csv}")
            for row in reader:
                existing_names.add(row["name"])
        write_header = False
        mode = "a"
        print(f"# generate: resuming {out_csv}; found {len(existing_names)} existing rows",
              flush=True)

    n_kept = 0
    n_failed = 0
    n_records = 0
    n_charge_match = 0
    start = time.time()

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["name", "representation"])

        for record_id, data, atoms, bonds in iter_sdf(sdf_path):
            n_records += 1
            if data.get("charge", "").strip() != charge_value:
                continue
            n_charge_match += 1
            if record_id in existing_names:
                continue

            if limit is not None and (n_kept + n_failed) >= limit:
                break

            try:
                rep = lalas_representation(atoms, bonds=bonds)
            except Exception as exc:
                n_failed += 1
                print(f"[fail] {record_id}: {exc}", flush=True)
                continue

            writer.writerow([record_id, rep])
            n_kept += 1

            if n_kept % 100 == 0:
                elapsed = time.time() - start
                rate = n_kept / elapsed if elapsed else 0.0
                print(f"... {n_kept} reps written  "
                      f"(records seen: {n_records}, charge-{charge_value}: {n_charge_match}, "
                      f"failed: {n_failed}, elapsed: {elapsed:.1f}s, rate: {rate:.1f}/s)",
                      flush=True)

    elapsed = time.time() - start
    print(f"# generate: wrote {n_kept} representations to {out_csv}", flush=True)
    print(f"#   records seen: {n_records}  "
          f"charge-{charge_value}: {n_charge_match}  "
          f"failed: {n_failed}  "
          f"elapsed: {elapsed:.1f}s", flush=True)
    return n_kept


# -------------------------------------------------- step 2: merge with feats

def merge_with_features(rep_csv: Path, features_csv: Path, out_csv: Path,
                        charge_value: str = "0") -> int:
    """
    Stream ``features_csv``, keep only rows whose ``charge`` column equals
    ``charge_value``, join with ``rep_csv`` on ``name`` and write
        name, formula, representation, <every other column from features_csv
        in its original order>
    to ``out_csv``. Rows in features that have no matching representation
    are skipped (and counted).
    """
    reps: dict[str, str] = {}
    with open(rep_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "name" not in reader.fieldnames \
                or "representation" not in reader.fieldnames:
            raise SystemExit(f"reps CSV missing required columns: {rep_csv}")
        for row in reader:
            reps[row["name"]] = row["representation"]
    print(f"# merge: loaded {len(reps)} representations from {rep_csv}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_matched = 0
    n_charge_rows = 0
    n_total_rows = 0
    n_no_rep = 0

    with open(features_csv, "r", encoding="utf-8", newline="") as f_in, \
            open(out_csv, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"features CSV is empty: {features_csv}")

        try:
            name_idx = header.index("name")
            charge_idx = header.index("charge")
            formula_idx = header.index("formula")
        except ValueError as exc:
            raise SystemExit(f"features CSV missing required column: {exc}")

        # Final header: name, formula, representation, then every other feature
        # column in its original order. ('charge' itself is kept in the output
        # so the downstream user can see it.)
        other_cols = [(i, h) for i, h in enumerate(header)
                      if i not in (name_idx, formula_idx)]
        out_header = ["name", "formula", "representation"] + [h for _, h in other_cols]

        writer = csv.writer(f_out)
        writer.writerow(out_header)

        max_needed = max(name_idx, charge_idx, formula_idx,
                         *(i for i, _ in other_cols))
        for row in reader:
            n_total_rows += 1
            if len(row) <= max_needed:
                continue
            if row[charge_idx].strip() != charge_value:
                continue
            n_charge_rows += 1

            name = row[name_idx]
            rep = reps.get(name)
            if rep is None:
                n_no_rep += 1
                continue

            writer.writerow(
                [name, row[formula_idx], rep] + [row[i] for i, _ in other_cols]
            )
            n_matched += 1

    print(f"# merge: wrote {n_matched} rows to {out_csv}", flush=True)
    print(f"#   features rows seen: {n_total_rows}  "
          f"charge-{charge_value}: {n_charge_rows}  "
          f"no-representation: {n_no_rep}", flush=True)
    return n_matched


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sdf", default=str(DEFAULT_SDF),
                        help=f"Input SDF (.sdf or .sdf.gz). Default: {DEFAULT_SDF}")
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES),
                        help=f"Feature CSV to merge against. Default: {DEFAULT_FEATURES}")
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR),
                        help=f"Output directory. Default: {DEFAULT_RESULTS_DIR}")
    parser.add_argument("--charge", default="0",
                        help="Charge value to keep (string match against the "
                             "SDF 'charge' data field and the CSV 'charge' "
                             "column). Default: 0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many charge-matching SDF records "
                             "(useful for testing).")
    parser.add_argument("--rep-name", default="compas-2x_lalas.csv",
                        help="Filename for the representations CSV.")
    parser.add_argument("--final-name", default="compas-2x_lalas_features.csv",
                        help="Filename for the merged CSV.")
    parser.add_argument("--skip-reps", action="store_true",
                        help="Skip representation generation; only merge an "
                             "existing reps CSV with the features CSV.")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Only generate the reps CSV; do not merge.")
    parser.add_argument("--resume", action="store_true",
                        help="Append missing representation rows to an existing "
                             "reps CSV instead of overwriting it.")
    args = parser.parse_args()

    sdf = Path(args.sdf).expanduser()
    features_csv = Path(args.features_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    rep_csv = out_dir / args.rep_name
    final_csv = out_dir / args.final_name

    if not args.skip_reps:
        if not sdf.is_file():
            raise SystemExit(f"SDF not found: {sdf}")
        generate_representations_csv(sdf, rep_csv,
                                     charge_value=args.charge,
                                     limit=args.limit,
                                     resume=args.resume)

    if not args.skip_merge:
        if not rep_csv.is_file():
            raise SystemExit(f"Representations CSV not found: {rep_csv} "
                             f"(re-run without --skip-reps to create it)")
        if not features_csv.is_file():
            raise SystemExit(f"Features CSV not found: {features_csv}")
        merge_with_features(rep_csv, features_csv, final_csv,
                            charge_value=args.charge)


if __name__ == "__main__":
    main()
