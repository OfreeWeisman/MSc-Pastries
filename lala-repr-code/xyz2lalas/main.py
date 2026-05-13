"""
Command-line entry point for the LALAS pipeline.

Reads a molecule from XYZ / Gaussian-input / SDF, runs the knot/path/LALAS
pipeline on it, and prints a single compact line per molecule:

    <ring types with inlined heteroatom positions> <augmented LALAS>

Every heteroatom-bearing ring carries its positions in brackets right
after its name, e.g. ``pyrazine[N:1,4]``, ``BH-diborinine[B:1,4]``, or
``ring-name[N:1,O:3]`` for a ring with mixed heteroatoms. Carbon-only
rings (benzene, cyclobutadiene, ...) have no positions and are rendered
without brackets.

For SDF inputs, ``--all-molecules`` or ``--first-k K`` can be used to
process multiple records in a single invocation.
"""

import sys
import argparse as ap

from . import molloader
from . import knotidentifier
from . import pathfinder
from . import chirality
from . import ringclassifier


__all__ = ['main']


COVALENCY_FACTOR = 1.3  # how much a covalent bond may exceed the sum of covalent radii


def main():
    args = parse_args()
    infile = args['infile']
    process_all = args['all_molecules']
    first_k = args['first_k']
    is_sdf = molloader.determine_filetype(infile) == 'sdf'

    if (process_all or first_k is not None) and is_sdf:
        for idx, molrepr in enumerate(molloader.iter_sdf(infile), start=1):
            if first_k is not None and idx > first_k:
                break
            process_molecule(molrepr)
    else:
        process_molecule(molloader.load(infile))

    sys.exit(0)


def process_molecule(molrepr, covalency_factor=COVALENCY_FACTOR):
    """Run the full pipeline on one Mol and print its final sequence line."""
    molrepr.align_to_xy_plane()

    knots, _ = knotidentifier.identify(molrepr, covalency_factor)
    assert len(knots) > 2, "\nNot enough knots found, need at least 3."

    paths, graph, knotedges = pathfinder.find(knots)

    print_final_sequence_line(paths, knotedges, knots, graph, molrepr.atoms)


def print_final_sequence_line(paths, edges, knots, graph, all_atoms,
                              covalency_factor=COVALENCY_FACTOR):
    """
    Print one compact line:

        <ring types with inlined heteroatom positions> <augmented LALAS>

    Ring classification, including each heteroatom-bearing ring's inlined
    positions (e.g. ``pyrazine[N:1,4]``, ``furan[O:3]``,
    ``BH-diborinine[B:1,4]``), is owned by
    ``ringclassifier.canonical_ring_sequence``. Carbon-only rings (benzene,
    cyclobutadiene, ...) have no heteroatom positions and render without
    brackets. Ring types and LALAS share the same canonical traversal, so
    they describe the same molecule in the same order.
    """
    names_by_index = {
        k.index: ringclassifier.classify_ring(k, all_atoms, covalency_factor)
        for k in knots
    }

    ring_types = ringclassifier.canonical_ring_sequence(
        paths, edges, knots, graph, all_atoms, covalency_factor)
    _, augmented_lalas, _, _ = chirality.canonical_lalas_fields(
        paths, edges, knots, graph, ring_type_for_idx=names_by_index.get)

    print(f"{ring_types} {augmented_lalas}")


def parse_args():
    parser = ap.ArgumentParser(
        description="Code to obtain LALAS representation for input PBH.")

    parser.add_argument(
        "infile",
        metavar="INFILE",
        help="Input file. Supported types: .xyz, .in (Gaussian input), .sdf",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--all-molecules",
        action="store_true",
        help="For SDF input, process every molecule record instead of only the first one.",
    )
    parser.add_argument(
        "--first-k",
        type=positive_int,
        default=None,
        metavar="K",
        help="For SDF input, process only the first K molecule records.",
    )

    return vars(parser.parse_args())


def positive_int(value):
    """argparse type ensuring strictly positive integer values."""
    parsed = int(value)
    if parsed <= 0:
        raise ap.ArgumentTypeError("K must be a positive integer")
    return parsed


if __name__ == '__main__':
    main()
