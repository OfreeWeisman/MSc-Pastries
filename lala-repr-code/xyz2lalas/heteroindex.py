"""
Heteroatom-position encoder for rings that contain at least one
heteroatom, in canonical LALAS traversal order.

Every heteroatom-bearing ring gets a token; only carbon-only rings (e.g.
benzene, cyclobutadiene) are absent from the output because they have no
heteroatom positions to report.

For each emitted ring, atoms are numbered ``0..(n-1)`` starting from one
of the fusion atoms with the ring's anchor (its parent in the canonical
tree, or the first other ring in the canonical sequence when this ring
is the canonical root). The two fusion atoms always end up at indices 0
and ``n-1``, so the shared edge sits at the wrap-around ``(n-1) -> 0``.

Atom 0 is chosen so that its direction-next neighbour in the cyclic
ordering is NOT the other fusion atom.

Numbering direction is the OPPOSITE of the chirality label - see the note
in ``canonical_heteroatom_tokens`` for why.

Output: per heteroatom-bearing ring, a single token of the form
``"ELEMENT:i,j[,ELEMENT2:k,...]"`` (elements sorted alphabetically,
positions ascending). Examples: ``"N:1,4"`` for pyrazine, ``"O:3"`` for
furan, ``"B:1,4"`` for a 1,4-(BH-)diborinine, and ``"N:1,O:3"`` for a
hypothetical ring with one N and one O. Tokens are returned keyed by
ring-knot index so each one can be inlined next to its ring in the
canonical ring-type sequence rendering.

Canonical traversal and chirality direction are both delegated to
``chirality.py``, so this module does not duplicate any LALAS
canonicalisation logic.
"""

import math

from . import chirality
from . import ringclassifier


__all__ = ['canonical_heteroatom_tokens']


def canonical_heteroatom_tokens(paths, edges, knots, graph, all_atoms,
                                covalency_factor=1.3):
    """
    Return ``{ring_knot_index: token}`` for every ring along the canonical
    traversal that contains at least one heteroatom. Carbon-only rings
    (e.g. benzene, cyclobutadiene) are absent from the dict because they
    have no heteroatom positions to report.

    Token format is ``"ELEMENT:i,j[,ELEMENT2:k,...]"`` (see module docstring).
    """
    names_by_index = {
        k.index: ringclassifier.classify_ring(k, all_atoms, covalency_factor)
        for k in knots
    }
    canonical, chir_label, _ = chirality.canonical_path_and_chirality(
        paths, edges, knots, graph,
        ring_type_for_idx=names_by_index.get,
        with_indices=True)
    if not canonical:
        return {}

    # LALAS's get_rotation uses cross(a-b, c-b), the opposite sign of the
    # standard turn-at-b cross product cross(b-a, c-b). Net effect: LALAS
    # annulation 'A' (label "clockwise") corresponds to a math-CCW turn in
    # the aligned coords, and 'a' ("counter-clockwise") to a math-CW turn.
    # So the math direction we walk a ring in is the OPPOSITE of the
    # chirality label. Truly ambiguous (after ring-type tiebreak) keeps the
    # math-CW default.
    direction = 'CCW' if chir_label == 'clockwise' else 'CW'

    parent_map = _build_parent_map(canonical)
    knot_by_index = {k.index: k for k in knots}

    tokens = {}
    seen = set()
    for item in canonical:
        if not isinstance(item, int) or item in seen:
            continue
        seen.add(item)

        ring = knot_by_index[item]
        # The only rings we skip are carbon-only ones (benzene,
        # cyclobutadiene, ...) - they simply have no heteroatom positions
        # to emit. Every other ring contributes a token.
        if not any(a.element != 'C' for a in ring.atoms):
            continue

        anchor_idx = parent_map.get(item)
        if anchor_idx is None:
            anchor_idx = _first_other_ring(canonical, item)
        if anchor_idx is None:
            continue

        token = _encode_ring_heteroatoms(
            ring, knot_by_index[anchor_idx], direction)
        if token is not None:
            tokens[item] = token

    return tokens


def _encode_ring_heteroatoms(ring, anchor_ring, direction):
    """
    Number ``ring``'s atoms 0..n-1 in the requested cyclic ``direction``,
    starting from a fusion atom with ``anchor_ring`` so the shared edge
    lands at the wrap-around. Return a heteroatom token of the form
    ``"ELEMENT:i,j[,ELEMENT2:k,...]"`` (elements sorted alphabetically,
    positions ascending), or ``None`` if no valid atom 0 can be picked or
    the ring has no heteroatom.
    """
    ordered = _cyclic_order(ring.atoms, ring.x, ring.y, direction)
    fusion_idx_set = (
        {a.index for a in ring.atoms} & {a.index for a in anchor_ring.atoms}
    )
    fusion_atoms = [a for a in ring.atoms if a.index in fusion_idx_set]

    atom_zero = _pick_atom_zero(ordered, fusion_atoms)
    if atom_zero is None:
        return None

    start = ordered.index(atom_zero)
    rotated = ordered[start:] + ordered[:start]

    positions_by_element = {}
    for i, a in enumerate(rotated):
        if a.element != 'C':
            positions_by_element.setdefault(a.element, []).append(i)
    if not positions_by_element:
        return None

    return ",".join(
        f"{el}:{','.join(str(p) for p in sorted(positions))}"
        for el, positions in sorted(positions_by_element.items())
    )


def _build_parent_map(canonical):
    """
    Walk the canonical sequence (ints + '(' / ')' markers) and return a
    dict mapping each ring index to the ring it is fused with along the
    canonical tree (its "parent"). The root ring maps to ``None``.

    Inside a branch, every ring's parent is the branch knot (the ring just
    before the matching ``'('``). When the branch closes, the next
    main-chain ring's parent is also that branch knot - that's why we
    restore ``prev_ring`` from the stack on ``')'``.
    """
    parent = {}
    stack = []
    prev_ring = None
    for item in canonical:
        if item == '(':
            stack.append(prev_ring)
        elif item == ')':
            prev_ring = stack.pop()
        else:
            parent[item] = prev_ring
            prev_ring = item
    return parent


def _first_other_ring(canonical, this_idx):
    """Return the first ring index in ``canonical`` that is not ``this_idx``."""
    for item in canonical:
        if isinstance(item, int) and item != this_idx:
            return item
    return None


def _cyclic_order(ring_atoms, cx, cy, direction):
    """
    Sort ring atoms by polar angle around ``(cx, cy)``. ``atan2`` ascending
    gives CCW (math convention); reverse for CW.
    """
    by_angle = sorted(ring_atoms, key=lambda a: math.atan2(a.y - cy, a.x - cx))
    return list(reversed(by_angle)) if direction == 'CW' else by_angle


def _pick_atom_zero(ordered, fusion_atoms):
    """
    Return the fusion atom whose direction-next neighbour in ``ordered``
    is NOT the other fusion atom. That places the shared edge at the
    wrap-around ``(n-1) -> 0`` of the indexing.
    """
    fusion_idx_set = {a.index for a in fusion_atoms}
    n = len(ordered)
    for fa in fusion_atoms:
        i = ordered.index(fa)
        nxt = ordered[(i + 1) % n]
        if nxt.index not in fusion_idx_set:
            return fa
    return None
