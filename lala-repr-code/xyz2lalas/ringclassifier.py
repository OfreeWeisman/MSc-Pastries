"""
Classify the monocyclic ring building blocks of a polycyclic aromatic system.

Classification uses two layers:

1. **Topology** (always): ring size (4 / 5 / 6 atoms) and the sorted
   heteroatom inventory (counts of N, O, S, B in the ring). This alone
   distinguishes most ring types - benzene, pyridine, pyrazine, borinine,
   borole, pyrrole, furan, thiophene, cyclobutadiene.

2. **Geometry** (when ``all_atoms`` is provided): the only ring family that
   topology cannot distinguish is the 6-ring with two borons. When each B
   carries one extra H (an explicit B-H bond outside the ring) the ring
   is labelled ``'BH-diborinine'``. Otherwise - most commonly fused
   borons with 0 extra H - it stays as the generic ``'diborinine'``.

The label intentionally does not encode the boron positions inside the
ring (1,2- / 1,3- / 1,4-). Those positions are recovered by ``heteroindex``
as a separate per-ring token, so a `BH-diborinine` ring with B at 1 and
4 will appear as ``BH-diborinine`` plus the token ``"14"``.

Position isomers (e.g. 1,2- vs 1,4-diborinine, pyridazine vs pyrazine) are
not distinguished. Anything that does not match a known signature returns
``'unknown'``.

This module also renders the canonical ring-type sequence (with branch
parentheses) by reusing ``chirality.canonical_ring_indices`` to obtain the
canonical traversal order.
"""

from collections import Counter

from .const import __COV_RADII__


__all__ = [
    'classify_ring',
    'canonical_ring_sequence',
    'print_canonical_ring_sequence',
]


_RING_TYPES = {
    # (ring_size, heteroatom composition) -> name
    (6, ()):                    'Bn',
    (6, (('N', 1),)):           'Pd',
    (6, (('N', 2),)):           'Pz',
    (6, (('B', 1),)):           'Bz',
    (6, (('B', 2),)):           'DBn',  # generic; refined when geometry is given

    (5, (('B', 1),)):           'Bl',
    (5, (('N', 1),)):           'Py',
    (5, (('O', 1),)):           'Fu',
    (5, (('S', 1),)):           'Th',

    (4, ()):                    'Cbd',
}


def classify_ring(knot, all_atoms=None, covalency_factor=1.3):
    """
    Return the ring-type name for ``knot`` or ``'unknown'``.

    When ``all_atoms`` is provided and the ring is a 2-B 6-ring with
    each boron carrying exactly one attached hydrogen (an explicit B-H
    bond outside the ring), the label is refined to ``'BH-diborinine'``.
    Otherwise the generic ``'diborinine'`` label is returned.
    """
    size = len(knot.atoms)
    counts = Counter(a.element for a in knot.atoms if a.element != 'C')
    hetero_sig = tuple(sorted(counts.items()))
    base = _RING_TYPES.get((size, hetero_sig), 'unknown')

    if all_atoms is not None and size == 6 and counts.get('B', 0) == 2:
        return _classify_diborinine_variant(knot, all_atoms, covalency_factor)

    return base


def canonical_ring_sequence(paths, edges, knots, graph,
                            all_atoms=None, covalency_factor=1.3):
    """
    Return the canonical ring-type sequence as a string, with '(' and ')'
    around branches and no commas hugging the branch parentheses.

    Each heteroatom-bearing ring is rendered as ``"<name>[<positions>]"``
    (e.g. ``"pyrazine[N:1,4]"``, ``"BH-diborinine[B:1,4]"``); carbon-only
    rings (benzene, cyclobutadiene, ...) render plain. Heteroatom positions
    are produced by ``heteroindex.canonical_heteroatom_tokens`` so this
    function returns a fully self-contained ring-type description in one
    call.

    Pass ``all_atoms`` to enable B-H-aware diborinine disambiguation.
    """
    from . import chirality
    from . import heteroindex

    names_by_index = {
        k.index: classify_ring(k, all_atoms, covalency_factor) for k in knots
    }
    indices = chirality.canonical_ring_indices(
        paths, edges, knots, graph, ring_type_for_idx=names_by_index.get)
    decorations = heteroindex.canonical_heteroatom_tokens(
        paths, edges, knots, graph, all_atoms, covalency_factor)

    rendered = []
    for item in indices:
        if not isinstance(item, int):
            rendered.append(item)
            continue
        name = names_by_index[item]
        deco = decorations.get(item)
        rendered.append(f"{name}[{deco}]" if deco else name)

    return ",".join(rendered).replace("(,", "(").replace(",)", ")")


def print_canonical_ring_sequence(paths, edges, knots, graph,
                                  all_atoms=None, covalency_factor=1.3):
    """Print the canonical ring-type sequence (see ``canonical_ring_sequence``)."""
    print(canonical_ring_sequence(
        paths, edges, knots, graph, all_atoms, covalency_factor))


def _classify_diborinine_variant(knot, all_atoms, covalency_factor):
    """
    Refine the 2-B 6-ring label. Only two cases occur in practice:

    * each boron is bonded only to its two ring-carbon neighbours
      (0 attached H) -> generic ``'diborinine'``
    * each boron carries one explicit B-H bond outside the ring
      (1 attached H) -> ``'BH-diborinine'``
    """
    b_atoms = [a for a in knot.atoms if a.element == 'B']
    each_b_has_one_h = all(
        _count_attached_h(b, all_atoms, covalency_factor) == 1
        for b in b_atoms
    )
    return 'DhDBn' if each_b_has_one_h else 'DBn'


def _count_attached_h(ring_atom, all_atoms, covalency_factor):
    """
    Number of H atoms covalently bonded to ``ring_atom``, using the same
    covalent-radii cutoff as the rest of the pipeline.
    """
    h_radius = __COV_RADII__['H']
    base_radius = __COV_RADII__[ring_atom.element]
    cutoff_sq = ((base_radius + h_radius) * covalency_factor) ** 2

    n_h = 0
    for other in all_atoms:
        if other.element != 'H' or other.index == ring_atom.index:
            continue
        dx = ring_atom.x - other.x
        dy = ring_atom.y - other.y
        dz = ring_atom.z - other.z
        if dx * dx + dy * dy + dz * dz <= cutoff_sq:
            n_h += 1
    return n_h
