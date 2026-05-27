"""
Debug script for investigating heteroindex direction inconsistency.

Usage (from lala-repr-code/ directory, venv activated):
    python scripts/debug_molecule.py --name C2M409023
    python scripts/debug_molecule.py --name C2M409023 --sdf "path/to/compas-2x.sdf.gz"

Prints:
  - All rings with their atoms and bonded_atom_indices
  - Canonical path and chirality decision
  - For each heteroatom ring: cyclic order produced by _bond_cyclic_order,
    signed area, whether direction was reversed, and final token
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LALA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = LALA_DIR.parent

sys.path.insert(0, str(LALA_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from xyz2lalas import knotidentifier, pathfinder, chirality, ringclassifier, mol
from xyz2lalas import heteroindex as hetero_mod
from compas2lalas import iter_sdf

DEFAULT_SDF = PROJECT_ROOT / "examples" / "compas-2x.sdf"
COVALENCY_FACTOR = 1.3


# ------------------------------------------------------------------ helpers

def _signed_area(ordered):
    area = 0.0
    for i, atom in enumerate(ordered):
        nxt = ordered[(i + 1) % len(ordered)]
        area += atom.x * nxt.y - nxt.x * atom.y
    return area


def _bond_cyclic_order_verbose(ring_atoms, direction):
    """Like heteroindex._bond_cyclic_order but prints what it does."""
    ring_ids = {a.index for a in ring_atoms}
    atom_by_id = {a.index: a for a in ring_atoms}
    adjacency = {}
    for atom in ring_atoms:
        neighbours = [
            idx for idx in getattr(atom, 'bonded_atom_indices', ())
            if idx in ring_ids
        ]
        if len(neighbours) != 2:
            print(f"    [FALLBACK] atom {atom.index} has {len(neighbours)} ring-neighbours "
                  f"in bonded_atom_indices={getattr(atom,'bonded_atom_indices',None)}")
            return None, "fallback"
        adjacency[atom.index] = sorted(neighbours)

    start = min(ring_ids)
    prev = None
    current = start
    ordered_ids = []
    while True:
        ordered_ids.append(current)
        choices = [idx for idx in adjacency[current] if idx != prev]
        if not choices:
            return None, "no-choices"
        nxt = choices[0]
        if nxt == start:
            break
        if nxt in ordered_ids:
            return None, "cycle-break"
        prev, current = current, nxt
        if len(ordered_ids) > len(ring_atoms):
            return None, "overflow"

    if len(ordered_ids) != len(ring_atoms):
        return None, "length-mismatch"

    ordered = [atom_by_id[idx] for idx in ordered_ids]
    area = _signed_area(ordered)
    reversed_flag = False
    if (direction == 'CCW' and area < 0) or (direction == 'CW' and area > 0):
        ordered.reverse()
        reversed_flag = True

    print(f"    bond-walk order (pre-flip): {ordered_ids}")
    print(f"    signed area = {area:.4f}  direction={direction}  reversed={reversed_flag}")
    return ordered, "ok"


# ------------------------------------------------------------------ main

def debug_molecule(mol_name: str, sdf_path: Path):
    # --- find the record
    mol_name_lower = mol_name.lower()
    found = None
    for record_id, data, atoms_raw, bonds in iter_sdf(sdf_path):
        if record_id.lower() == mol_name_lower:
            found = (record_id, data, atoms_raw, bonds)
            break
    if found is None:
        print(f"Molecule '{mol_name}' not found in {sdf_path}")
        return

    record_id, data, atoms_raw, bonds = found
    print(f"=== {record_id}  charge={data.get('charge','?')}  "
          f"formula={data.get('formula','?')} ===\n")

    # --- build Mol and align
    rows = [[el, x, y, z] for (el, x, y, z) in atoms_raw]
    molrepr = mol.Mol(rows, bonds)
    molrepr.align_to_xy_plane()

    # --- identify rings
    knots, _ = knotidentifier.identify(molrepr, COVALENCY_FACTOR)
    if len(knots) < 3:
        print(f"Only {len(knots)} rings found; need ≥3.")
        return

    paths, graph, knotedges = pathfinder.find(knots)

    names_by_index = {
        k.index: ringclassifier.classify_ring(k, molrepr.atoms, COVALENCY_FACTOR)
        for k in knots
    }

    # --- print all rings
    print(f"--- Rings ({len(knots)} total) ---")
    for k in knots:
        atom_list = ", ".join(
            f"{a.index}:{a.element}({a.x:.3f},{a.y:.3f})" for a in k.atoms
        )
        print(f"  Ring {k.index} [{names_by_index[k.index]}]  center=({k.x:.3f},{k.y:.3f})")
        print(f"    atoms: {atom_list}")
        for a in k.atoms:
            bi = getattr(a, 'bonded_atom_indices', None)
            ring_bi = {i for i in (bi or set()) if any(aa.index == i for aa in k.atoms)}
            print(f"      atom {a.index}:{a.element}  bonded_atom_indices={bi}  ring_neighbours={ring_bi}")
    print()

    # --- chirality decision
    canonical_indices, chir_label, match_type = chirality.canonical_path_and_chirality(
        paths, knotedges, knots, graph,
        ring_type_for_idx=names_by_index.get,
        with_indices=True)

    print(f"--- Chirality ---")
    print(f"  label={chir_label}  match_type={match_type}")
    print(f"  canonical ring order: {canonical_indices}")
    print()

    # --- direction used for heteroindex (matches chirality label)
    direction = 'CW' if chir_label == 'clockwise' else 'CCW'
    print(f"  -> heteroindex direction = '{direction}'  "
          f"(matches chirality label)\n")

    # --- trace each heteroatom ring
    parent_map = hetero_mod._build_parent_map(canonical_indices)
    knot_by_index = {k.index: k for k in knots}

    print("--- Heteroatom ring traversal ---")
    seen = set()
    for item in canonical_indices:
        if not isinstance(item, int) or item in seen:
            continue
        seen.add(item)

        ring = knot_by_index[item]
        has_hetero = any(a.element != 'C' for a in ring.atoms)
        ring_label = names_by_index[item]
        print(f"\n  Ring {item} [{ring_label}]  has_hetero={has_hetero}")

        if not has_hetero:
            print(f"    -> skipped (carbon-only)")
            continue

        anchor_idx = parent_map.get(item)
        if anchor_idx is None:
            anchor_idx = hetero_mod._first_other_ring(canonical_indices, item)
        print(f"    anchor ring = {anchor_idx}")

        if anchor_idx is None:
            print("    -> no anchor found, skipping")
            continue

        anchor_ring = knot_by_index[anchor_idx]
        fusion_idx_set = (
            {a.index for a in ring.atoms} & {a.index for a in anchor_ring.atoms}
        )
        fusion_atoms = [a for a in ring.atoms if a.index in fusion_idx_set]
        print(f"    fusion atoms: {[(a.index, a.element) for a in fusion_atoms]}")

        # Verbose bond cyclic order
        ordered, status = _bond_cyclic_order_verbose(ring.atoms, direction)

        if ordered is None:
            print(f"    -> _bond_cyclic_order returned None ({status}), using polar-angle fallback")
            by_angle = sorted(ring.atoms, key=lambda a: math.atan2(a.y - ring.y, a.x - ring.x))
            ordered = list(reversed(by_angle)) if direction == 'CW' else by_angle
            area = _signed_area(ordered)
            print(f"    fallback polar-angle order: {[a.index for a in ordered]}  area={area:.4f}")
        else:
            print(f"    final cyclic order: {[a.index for a in ordered]}")

        atom_zero = hetero_mod._pick_atom_zero(ordered, fusion_atoms)
        if atom_zero is None:
            print("    -> _pick_atom_zero returned None")
            continue
        print(f"    atom_zero = {atom_zero.index}:{atom_zero.element}")

        start = ordered.index(atom_zero)
        rotated = ordered[start:] + ordered[:start]
        print(f"    rotated sequence: {[(a.index, a.element) for a in rotated]}")

        for i, a in enumerate(rotated):
            tag = f"  <- heteroatom at {i}" if a.element != 'C' else ""
            print(f"      [{i}] {a.index}:{a.element}{tag}")

        # compute token
        positions_by_element: dict[str, list[int]] = {}
        for i, a in enumerate(rotated):
            if a.element != 'C':
                positions_by_element.setdefault(a.element, []).append(i)

        token = ",".join(
            f"{el}:{','.join(str(p) for p in sorted(pos))}"
            for el, pos in sorted(positions_by_element.items())
        ) if positions_by_element else None
        print(f"    => token: {token}")

    # --- final representation
    print("\n--- Final representation ---")
    ring_types = ringclassifier.canonical_ring_sequence(
        paths, knotedges, knots, graph, molrepr.atoms, COVALENCY_FACTOR)
    _, augmented_lalas, _, _ = chirality.canonical_lalas_fields(
        paths, knotedges, knots, graph, ring_type_for_idx=names_by_index.get)
    print(f"  ring_types:      {ring_types}")
    print(f"  augmented_lalas: {augmented_lalas}")
    print(f"  full repr:       {ring_types} {augmented_lalas}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True,
                        help="Molecule name to look up in the SDF (case-insensitive).")
    parser.add_argument("--sdf", default=str(DEFAULT_SDF),
                        help=f"SDF file to search. Default: {DEFAULT_SDF}")
    args = parser.parse_args()

    sdf_path = Path(args.sdf).expanduser().resolve()
    if not sdf_path.is_file():
        raise SystemExit(f"SDF not found: {sdf_path}")

    debug_molecule(args.name, sdf_path)


if __name__ == "__main__":
    main()
