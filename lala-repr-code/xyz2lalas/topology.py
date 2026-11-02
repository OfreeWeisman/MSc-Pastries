"""
Annulation and ring orientation from the bond graph alone.

Every ring gets a cyclic atom order, and all fused rings share one
orientation: two rings fused at bond (u, v) traverse it in opposite
directions, as two faces of an oriented surface do. The orientation is
propagated from one ring through the shared bonds, so it is defined by the
connectivity, not by coordinates. Which of the two global orientations is
picked is arbitrary; the LALAS canonicalisation already considers both (it
compares every sequence with its inversion), so the output does not depend
on it.

The annulation at a ring ``b`` between neighbours ``a`` and ``c`` follows
from where ``b`` is fused to them. With ``n`` bonds in ``b`` and ``k`` the
number of steps, along the orientation, from the bond shared with ``a`` to
the bond shared with ``c``:

* ``2k == n`` - the exit bond is opposite the entry bond: linear, ``0``
  (``L``). Only even rings have an opposite bond, so a 6-ring can be
  linear or angular, a 4-ring is always linear and a 5-ring always turns.
* ``2k < n`` / ``2k > n`` - angular, ``2`` (``a``) / ``1`` (``A``).

Only the ring size enters, never the ring type, so any 4-, 5- or 6-ring
(and in principle any size) is handled the same way.

For a planar molecule whose orientation happens to be counter-clockwise in
the xy-plane, these values equal the former geometric rule (angle between
ring centres, sign of the turn in the plane), and numbering 'CCW' runs
along the orientation. Bent or non-planar molecules no longer depend on how
they project onto a plane.
"""

from collections import deque


__all__ = ['ring_orientation', 'turn', 'oriented_atoms']


def ring_orientation(knots):
    """
    Return ``{knot_index: [atom indices in cyclic order]}`` with one
    consistent orientation for every set of fused rings.
    """
    cycles = {k.index: _ring_cycle(k) for k in knots}
    atom_sets = {k.index: {a.index for a in k.atoms} for k in knots}

    orientation = {}
    for root in sorted(cycles):
        if root in orientation:
            continue
        orientation[root] = cycles[root]
        queue = deque([root])
        while queue:
            ring = queue.popleft()
            for other in sorted(cycles):
                if other in orientation:
                    continue
                shared = atom_sets[ring] & atom_sets[other]
                if len(shared) != 2:
                    continue
                u, v = _directed_edge(orientation[ring], shared)
                # The neighbour must run through the shared bond as v -> u.
                cycle = cycles[other]
                if _directed_edge(cycle, shared) == (u, v):
                    cycle = cycle[::-1]
                orientation[other] = cycle
                queue.append(other)
    return orientation


def turn(orientation, prev_idx, cur_idx, next_idx):
    """Annulation at ring ``cur_idx`` between its neighbours: 0 (L), 1 (A) or 2 (a)."""
    cycle = orientation[cur_idx]
    n = len(cycle)
    k = (_edge_position(cycle, orientation[next_idx])
         - _edge_position(cycle, orientation[prev_idx])) % n
    if 2 * k == n:
        return 0
    return 2 if 2 * k < n else 1


def oriented_atoms(orientation, knot, direction):
    """
    ``knot``'s atoms in cyclic order: along the orientation for 'CCW',
    against it for 'CW'.
    """
    by_index = {a.index: a for a in knot.atoms}
    ordered = [by_index[i] for i in orientation[knot.index]]
    return ordered[::-1] if direction == 'CW' else ordered


def _ring_cycle(knot):
    """Atom indices of ``knot`` in cyclic order, following the bonds."""
    ring_ids = {a.index for a in knot.atoms}
    neighbours = {}
    for atom in knot.atoms:
        inside = sorted(i for i in getattr(atom, 'bonded_atom_indices', ()) if i in ring_ids)
        if len(inside) != 2:
            raise ValueError(
                f"ring atom {atom.index} has {len(inside)} bonds inside its ring; "
                "need exactly 2 to order the ring (are bonds missing?)")
        neighbours[atom.index] = inside

    start = min(ring_ids)
    cycle = [start]
    previous = None
    while True:
        current = cycle[-1]
        nxt = next(i for i in neighbours[current] if i != previous)
        if nxt == start:
            break
        if nxt in cycle:
            raise ValueError(f"ring {knot.index} is not a simple cycle")
        cycle.append(nxt)
        previous = current
    if len(cycle) != len(ring_ids):
        raise ValueError(f"ring {knot.index} is not a simple cycle")
    return cycle


def _directed_edge(cycle, shared):
    """The shared bond as it is traversed in ``cycle``: ``(u, v)``, u first."""
    n = len(cycle)
    for i in range(n):
        u, v = cycle[i], cycle[(i + 1) % n]
        if {u, v} == shared:
            return u, v
    raise ValueError("rings share two atoms that are not bonded in the ring")


def _edge_position(cycle, other_cycle):
    """Position ``i`` of the bond (cycle[i], cycle[i+1]) shared with ``other_cycle``."""
    shared = set(cycle) & set(other_cycle)
    n = len(cycle)
    for i in range(n):
        if {cycle[i], cycle[(i + 1) % n]} == shared:
            return i
    raise ValueError("rings are not fused at a bond")
