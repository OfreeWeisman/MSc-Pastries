import numpy as np
import networkx as nx


from . import knot
from . import mol
from . import molloader

from .const import __COV_RADII__


# Maximum number of heavy-atom (non-H) neighbours a given element is
# allowed to have. Used to prune spurious long-distance "bonds" that the
# pure covalent-radii cutoff would otherwise accept (e.g. a 2.05 A B-S
# contact next to an already 2-coordinated thiophene S). The caps cover
# the typical aromatic/sp2/sp3 chemistries this code is used for; any
# element not listed falls back to ``_DEFAULT_MAX_HEAVY`` (4), which is
# generous enough to accept any reasonable organic geometry.
_MAX_HEAVY_NEIGHBOURS = {
    'B':  3,    # sp2 boron in boroles / borinines / diborinines (+ optional H)
    'C':  4,    # sp3 max; sp2 uses 3 anyway
    'N':  3,    # pyridinic / pyrrolic N
    'O':  2,    # furan / ether / alcohol O
    'F':  1,
    'Si': 4,
    'P':  4,    # phosphine / phosphate (no expanded octets in this dataset)
    'S':  2,    # thiophene S; sulfones (>2) are out of scope here
    'Cl': 1,
    'Br': 1,
    'I':  1,
}
_DEFAULT_MAX_HEAVY = 4


def identify(_molrepr, covalency_factor):
    """
    identify(_molrepr: Mol, covalency_factor: float) -> knot: list(Knots), edges: list(tuples)

    Identify the rings in the molecule and initialize them as Knot Object.

    Ring perception uses ``nx.minimum_cycle_basis`` (smallest cycles in the
    cycle space, so fused polycyclic systems do not yield large composite
    cycles such as a 12-atom "ring" spanning multiple chemical rings).

    Bond source priority:
      1. If ``_molrepr.bonds`` is populated (typically from an SDF bond
         block), it is used verbatim - the molecule's authoritative
         chemistry is trusted and no geometric inference runs.
      2. Otherwise (xyz / Gaussian input - no explicit bonds in the file),
         bonds are inferred geometrically from covalent radii and then
         pruned by per-element heavy-neighbour caps: each atom accepts
         only its closest candidate neighbours up to its element-typical
         maximum coordination, and a bond survives only if both endpoints
         accept it. This removes spurious long-distance contacts (e.g. a
         B-S contact at the edge of the covalent-radii cutoff) without
         forcing a tighter global factor.

    in:
    _molrepr: A molecule, represented by a list of Atoms with their xyz coordinates in Angstroms.
    covalency_factor: A bond is identified if the sum of covelent radii times the covalency factor is larger than the distance between atoms. Only used when ``_molrepr.bonds`` is empty.

    out:
    knots: A list of Knots. Knots represent monocycles in this program.
    edges: A list of tuples. each tuple is a pair of atom (index) forming a bond in the molecule.

    """
    explicit_bonds = getattr(_molrepr, 'bonds', None)
    if explicit_bonds:
        edges = list(explicit_bonds)
    else:
        atom_connectivity = get_connectivity_matrix(_molrepr.atoms, covalency_factor, skip_hydrogen=False)
        edges = get_edges(atom_connectivity)
        edges = _prune_bonds_by_valence(edges, _molrepr.atoms)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(_molrepr.atoms))) # keep isolated atoms in the graph so node indices stay aligned
    graph.add_edges_from(edges)
    cycles = nx.minimum_cycle_basis(graph) # smallest fundamental rings (chemistry-natural)
    knots = get_knots(_molrepr.atoms, cycles) # monocycle = Knot. Knot contains all the atoms in the monocycle and index, type, geometrical center

    return knots, edges


def _prune_bonds_by_valence(edges, atoms):
    """
    Drop bonds that would over-coordinate an atom relative to its element's
    typical maximum number of heavy-atom neighbours.

    Algorithm:
      * For each atom, rank its candidate non-H heavy neighbours by distance.
      * Each atom "accepts" only its closest ``cap`` neighbours, where ``cap``
        is element-specific (``_MAX_HEAVY_NEIGHBOURS``; ``_DEFAULT_MAX_HEAVY``
        otherwise).
      * A bond is kept iff both endpoints accept the other (so a chemistry
        valid bond at element A only survives if it's also among element B's
        closest contacts).

    H-X bonds are passed through unchanged; H is always a leaf and never
    causes the spurious-bond problems this prune is designed to fix.
    """
    if not edges:
        return edges

    candidates_by_atom: dict[int, list[tuple[float, int]]] = {}
    for i, j in edges:
        if atoms[i].element == 'H' or atoms[j].element == 'H':
            continue
        dx = atoms[i].x - atoms[j].x
        dy = atoms[i].y - atoms[j].y
        dz = atoms[i].z - atoms[j].z
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        candidates_by_atom.setdefault(i, []).append((d, j))
        candidates_by_atom.setdefault(j, []).append((d, i))

    accepted: dict[int, set[int]] = {}
    for atom_idx, neighbours in candidates_by_atom.items():
        cap = _MAX_HEAVY_NEIGHBOURS.get(atoms[atom_idx].element, _DEFAULT_MAX_HEAVY)
        neighbours.sort()
        accepted[atom_idx] = {nbr for _, nbr in neighbours[:cap]}

    pruned = []
    for i, j in edges:
        if atoms[i].element == 'H' or atoms[j].element == 'H':
            pruned.append((i, j))
            continue
        if j in accepted.get(i, ()) and i in accepted.get(j, ()):
            pruned.append((i, j))

    return pruned


def get_connectivity_matrix(_atoms, covalency_factor, skip_hydrogen = False):
    """
    get_connectivity_matrix(_atoms: list(Atoms), covalency_factor: float, skip_hydrogen: bool = False) -> numpy.ndarray

    Function that loops through the atoms and returns the connectivity matrix. Two atoms are considered bonded when the distance between them is less 
    or equal to the sum of their covalent radii multiplied by a covalency factor. 

    in:
    _atoms: A list of Atoms with their xyz coordinates in Angstroms.
    covalency_factor: A bond is identified if the sum of covelent radii times the covalency factor is larger than the distance between atoms.
    skip_hydrogen: If True, remove hydrogens completely.

    out:
    connectivity_matrix: A connectivity matrix of dimension len(_atoms) x len(_atoms) where elements are 0 if there is no bond and 1 if there is a bond. 
                         Diagonal elements are 0.
    
    """
    number_of_atoms = len(_atoms)
    connectivity_matrix = np.zeros((number_of_atoms,number_of_atoms), dtype = int) # initialize matrix with 0s 

    for i in range(number_of_atoms):
        for j in range(i+1, number_of_atoms): # start at i+1 because diagonal elements should stay 0
            if skip_hydrogen: # skip hydrogens if set
                if _atoms[i].element == 'H' or _atoms[j].element == 'H':
                    continue
            covalency_cutoff = (__COV_RADII__[_atoms[i].element] + __COV_RADII__[_atoms[j].element]) * covalency_factor # determine cutoff for elements i,j
            distance_ij = np.sqrt(
                (_atoms[i].x - _atoms[j].x)**2 +
                (_atoms[i].y - _atoms[j].y)**2 +
                (_atoms[i].z - _atoms[j].z)**2
            )
            if distance_ij <= covalency_cutoff:
                connectivity_matrix[i,j] = connectivity_matrix[j,i] = 1

    return connectivity_matrix


def get_edges(_atom_connectivity):
    """
    get_edges(_atom_connectivity: numpy.ndarray) -> list(tuple)

    Using the connectivity matrix, this function generates a list of tuple, where every tuple contains the atomic index
    of two atoms bonding.

    in: 
    _atom_connectivity: Connectivity matrix.

    out:
    edges: A list of tuples that represent connections in the connectivity matrix, i.e. bonds in the molecule.

    """

    dimension = _atom_connectivity.shape[0]
    edges = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            if _atom_connectivity[i,j] == 1:
                edges.append((i,j))
    
    return edges


def get_knots(_atoms, _cycles):
    """
    get_knots(_atoms: list(Atom), _cycles: list(list(int)), _knot_types: dict) -> list(Knot)

    Function that gets the geometric center of each ring of the molecule and initializes the Knot Objects for each monocycle.

    in:
    _atoms: A list of Atoms with their xyz coordinates in Angstroms.
    _cycles: A list of monocycles. Each monocycle is a list of atom indices.
    _knot_types: A dict of monocyle types to look for.

    out:
    knots: A list of Knots (= monocycles).

    """

    knots = [] # initialize list to return
    i = 0
    for cycle in _cycles:
        cycle_atoms = ''
        x_knot = y_knot = z_knot = 0
        for atom in cycle:
            cycle_atoms += _atoms[atom].element
            x_knot += _atoms[atom].x
            y_knot += _atoms[atom].y
            z_knot += _atoms[atom].z
        
        knot_type = 'bn'
        _knot = knot.Knot(i, knot_type, [_atoms[x] for x in cycle], x_knot/len(cycle), y_knot/len(cycle), z_knot/len(cycle))
        i += 1
        knots.append(_knot)
    
    return knots
        
