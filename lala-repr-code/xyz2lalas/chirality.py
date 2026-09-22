"""
Chirality classifier for the canonical LALAS sequence.

Given the same inputs used by ``getlalas.get_LALAS`` (paths, edges, knots,
graph), this module decides whether the FIRST angular annulation of the
canonical sequence was clockwise or counter-clockwise in the molecule's
geometry. The canonical selection itself is reused from getlalas - this
module never alters which sequence is canonical, only which traversal
direction it corresponds to.

Possible labels:
  'clockwise'         The canonical equals one of the raw (pre-inversion)
                      sequences; its leading 'A' came from a rotation sign
                      < 0 in 3D.
  'counter-clockwise' The canonical equals an inverted raw sequence; its
                      leading 'A' is the mirror of an 'a' that came from a
                      rotation sign > 0 in 3D.
  'ambiguous'         Both a direct and an inverted raw sequence match the
                      canonical AND - if a ring-type lookup is provided -
                      the ring-type sequences along the two directions are
                      identical (e.g. phenanthrene's C2v symmetry).
  'none'              The canonical has no angular annulations (pure linear
                      polycyclic benzenoid).

The canonical LALAS sequence is fixed first (by ``getlalas``). When more
than one traversal reads as that sequence and ``ring_type_for_idx`` is
supplied, the traversal whose rendered representation (ring types with
heteroatom positions, compared token by token) is lex-smallest wins, and
its match type ('direct' -> 'clockwise', 'inverted' -> 'counter-clockwise')
becomes the effective chirality.
"""

from . import getlalas


__all__ = [
    'classify',
    'get_LALAS_with_chirality',
    'canonical_ring_indices',
    'canonical_lalas_fields',
    'canonical_path_and_chirality',
]


def classify(_paths, _edges, _knots, _graph, ring_type_for_idx=None):
    """Return the chirality label for the canonical LALAS sequence."""
    label, _, _ = canonical_path_and_chirality(
        _paths, _edges, _knots, _graph, ring_type_for_idx=ring_type_for_idx,
        with_indices=False)
    return label


def get_LALAS_with_chirality(_paths, _edges, _knots, _graph,
                             ring_type_for_idx=None):
    """
    Print one CSV line: ``balaban,augmented_lalas,lalas,chirality``.
    Mirrors ``getlalas.get_LALAS`` and appends the chirality label.
    """
    balaban_name, augmented_lalas, lalas, label = canonical_lalas_fields(
        _paths, _edges, _knots, _graph, ring_type_for_idx=ring_type_for_idx)
    print(f'{balaban_name},{augmented_lalas},{lalas},{label}')


def canonical_lalas_fields(_paths, _edges, _knots, _graph, ring_type_for_idx=None):
    """
    Return ``(balaban_name, augmented_lalas, lalas, chirality_label)`` for
    the canonical sequence.
    """
    final_seq, raw_sequences, raw_trees, raw_paths_idx = getlalas.compute_canonical_sequence(
        _paths, _edges, _knots, _graph)
    render = _make_renderer(raw_paths_idx, raw_trees, _knots, ring_type_for_idx)
    label, _, _ = _resolve(
        final_seq, raw_sequences, raw_paths_idx, ring_type_for_idx, render)
    balaban_name, augmented_lalas, lalas = getlalas.format_lalas_strings(final_seq)
    return balaban_name, augmented_lalas, lalas, label


def canonical_ring_indices(_paths, _edges, _knots, _graph,
                           ring_type_for_idx=None):
    """
    Return the canonical LALAS traversal as a list of knot indices
    interspersed with '(' and ')' for branches. The traversal direction
    follows the resolved chirality (direct match walks the original
    geometric path; inverted match walks the mirror path) so downstream
    callers (heteroindex, ringclassifier) all see the same direction.
    """
    indices, _, _ = canonical_path_and_chirality(
        _paths, _edges, _knots, _graph, ring_type_for_idx=ring_type_for_idx,
        with_indices=True)
    return indices


def canonical_path_and_chirality(_paths, _edges, _knots, _graph,
                                 ring_type_for_idx=None, with_indices=True):
    """
    Compute the canonical sequence once and resolve both the chirality label
    and (optionally) the canonical knot-index traversal in a single pass.

    Returns ``(indices, label, match_type)`` where ``indices`` is empty when
    ``with_indices`` is False or no canonical match is found.
    """
    final_seq, raw_sequences, raw_trees, raw_paths_idx = getlalas.compute_canonical_sequence(
        _paths, _edges, _knots, _graph)
    render = _make_renderer(raw_paths_idx, raw_trees, _knots, ring_type_for_idx)
    label, chosen_idx, match_type = _resolve(
        final_seq, raw_sequences, raw_paths_idx, ring_type_for_idx, render)

    if not with_indices or chosen_idx is None:
        return [], label, match_type

    indices = _get_index_sequence(raw_paths_idx[chosen_idx], raw_trees[chosen_idx])
    return indices, label, match_type


def _resolve(final_seq, raw_sequences, raw_paths_idx, ring_type_for_idx,
             render=None):
    """
    Decide which raw sequence represents the canonical traversal and
    return ``(label, chosen_idx, match_type)``.

    ``match_type`` is 'direct' when ``canonical == raw`` and 'inverted' when
    ``canonical == invert(raw)``. Every match already reads as the canonical
    LALAS sequence; among them the one whose ``render(idx, label)`` is
    lex-smallest wins (the first match when there is no renderer).
    """
    has_angular = _has_angular_annulation(final_seq)
    direct, inverted = _find_matches(final_seq, raw_sequences, raw_paths_idx)

    if not direct and not inverted:
        return ('none' if not has_angular else 'unknown'), None, None

    label_for = _label_rule(has_angular, direct, inverted, ring_type_for_idx)
    candidates = [(i, 'direct') for i, _ in direct] + [(i, 'inverted') for i, _ in inverted]
    chosen_idx, match_type = candidates[0]
    if render is not None:
        chosen_idx, match_type = min(
            candidates, key=lambda c: render(c[0], label_for(c[1])))
    return label_for(match_type), chosen_idx, match_type


def _label_rule(has_angular, direct, inverted, ring_type_for_idx):
    """
    Return ``label_for(match_type)``: the chirality label implied by
    choosing a candidate of that match type. A direct/inverted tie is
    'ambiguous' when there is no ring-type lookup or when both directions
    read as the same ring-type sequences (e.g. phenanthrene's C2v symmetry).
    """
    if not has_angular:
        return lambda _match_type: 'none'
    if direct and inverted and (
            ring_type_for_idx is None
            or _ring_seqs(direct, ring_type_for_idx) == _ring_seqs(inverted, ring_type_for_idx)):
        return lambda _match_type: 'ambiguous'
    return lambda match_type: 'clockwise' if match_type == 'direct' else 'counter-clockwise'


def _ring_seqs(matches, ring_type_for_idx):
    return sorted(_ring_seq(p, ring_type_for_idx) for _, p in matches)


def _has_angular_annulation(final_seq):
    return any(isinstance(v, int) and v in (1, 2) for v in final_seq)


def _find_matches(final_seq, raw_sequences, raw_paths_idx):
    """Return ``(direct_matches, inverted_matches)`` as ``(idx, path)`` lists."""
    direct = [(i, raw_paths_idx[i])
              for i, raw in enumerate(raw_sequences) if raw == final_seq]
    inverted = [(i, raw_paths_idx[i])
                for i, raw in enumerate(raw_sequences) if _invert(raw) == final_seq]
    return direct, inverted


def _make_renderer(raw_paths_idx, raw_trees, knots, ring_type_for_idx):
    """
    Return ``render(idx, label)``: the representation tokens (ring types
    with heteroatom positions, plus '(' / ')') that raw candidate ``idx``
    would produce under chirality ``label``, as a tuple so candidates
    compare token by token. ``None`` when there is no ring-type lookup.
    """
    if ring_type_for_idx is None:
        return None
    # Local import: heteroindex imports this module at load time.
    from . import heteroindex, ringclassifier

    def render(idx, label):
        indices = _get_index_sequence(raw_paths_idx[idx], raw_trees[idx])
        positions = heteroindex.tokens_for_traversal(indices, knots, label)
        return tuple(ringclassifier.ring_sequence_tokens(indices, ring_type_for_idx, positions))

    return render


def _ring_seq(path, ring_type_for_idx):
    return tuple(ring_type_for_idx(k) for k in path)


def _get_index_sequence(path, current_node):
    """
    Walk the same tree as ``getlalas.get_sequence`` but emit knot indices
    for every visited node (root, linear chain, branch point, leaves) and
    '(' / ')' around branches. The branch-direction choice mirrors
    ``get_sequence`` so the output order is parallel to the LALAS canonical
    sequence.
    """
    sequence = [current_node.index]

    while current_node.count_children() == 1:
        current_node = current_node.children[0]
        sequence.append(current_node.index)

    if current_node.count_children() > 1:
        sequence.append("(")
        if current_node.children[1].index in path:
            sequence.extend(_get_index_sequence(path, current_node.children[0]))
            sequence.append(")")
            sequence.extend(_get_index_sequence(path, current_node.children[1]))
        else:
            sequence.extend(_get_index_sequence(path, current_node.children[1]))
            sequence.append(")")
            sequence.extend(_get_index_sequence(path, current_node.children[0]))

    return sequence


def _invert(seq):
    """Swap angular annulations 1 <-> 2; leave 'L' (0) and brackets alone."""
    return [1 if s == 2 else 2 if s == 1 else s for s in seq]
