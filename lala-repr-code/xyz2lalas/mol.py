import math
import numpy as np

from dataclasses import dataclass


class Mol:
    def __init__(self, _atoms, _bonds=None):
        """
        ``_atoms`` is a list of ``[element, x, y, z]`` rows.

        ``_bonds`` is an optional list of ``(i, j)`` atom-index pairs
        (0-based). When provided (typically from an SDF bond block), the
        bond graph is taken as-is and no geometric inference is needed
        downstream. ``None`` means "no explicit bonds available - infer
        from geometry."
        """
        self.atoms = []
        i = 0
        for atom in _atoms:
            self.atoms.append(Atom(i, atom[0], atom[1], atom[2], atom[3]))
            i += 1
        self.bonds = _normalise_bonds(_bonds) if _bonds else None
        for atom in self.atoms:
            atom.bonded_atom_indices = set()
        if self.bonds:
            for i, j in self.bonds:
                self.atoms[i].bonded_atom_indices.add(j)
                self.atoms[j].bonded_atom_indices.add(i)


    def __str__(self):
        length = math.floor(math.log10(len(self.atoms))) + 1
        return "\n".join(f"{atom.index:{length}}  {atom.element:2} {atom.x:8.5f} {atom.y:8.5f} {atom.z:8.5f}" for atom in self.atoms)


    def __getitem__(self, index):
        return self.atoms[index]


    def align_to_xy_plane(self):
        """
        Rotate the molecule into xy-plane. In-place operation.

        """

        # Centroid (unit masses, matching the inertia tensor below). The
        # tensor must be taken about it: about the origin, the principal
        # axes - and so the projection used for every annulation angle -
        # would depend on where the molecule happens to sit in the input
        # frame, and a plain translation could change the representation.
        com = np.mean([[atom.x, atom.y, atom.z] for atom in self.atoms], axis=0)

        I = np.zeros((3,3)) # set up inertia tensor I about the centroid
        for atom in self.atoms:
            x, y, z = atom.x - com[0], atom.y - com[1], atom.z - com[2]
            I += np.array([[(y**2+z**2), -x*y       , -x*z       ],
                           [-x*y       , (x**2+z**2), -y*z       ],
                           [-x*z       , -y*z       , x**2+y**2  ]])

        # extract eigenvalues and eigenvectors for I
        # np.linalg.eigh(I)[0] are eigenValues, [1] are eigenVectors
        eigenVectors = np.linalg.eigh(I)[1]
        eigenVectorsTransposed = np.transpose(eigenVectors)

        # transform xyz to new coordinate system.
        # transform v1 -> ex, v2 -> ey, v3 -> ez
        for atom in self.atoms:
            xyz = [atom.x, atom.y, atom.z]
            atom.x, atom.y, atom.z = np.dot(eigenVectorsTransposed,xyz-com)


def _normalise_bonds(bonds):
    """Sort each (i, j) so i < j, drop self-loops, and dedupe while
    preserving order. Returns a fresh list."""
    seen = set()
    out = []
    for pair in bonds:
        a, b = int(pair[0]), int(pair[1])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


@dataclass
class Atom:
    index: int
    element: str
    x: float
    y: float
    z: float


    def __repr__(self):
        return f"{self.index:4} {self.element:2} {self.x:8.5f} {self.y:8.5f} {self.z:8.5f}"


    def __hash__(self):
        return hash(f"{self.index}{self.element}{self.x}{self.y}{self.z}")


    def __eq__(self, other):
        return self.index == other.index and self.element == other.element and math.isclose(self.x, other.x, 1e-9, 1e-9) and math.isclose(self.y, other.y, 1e-9, 1e-9) and math.isclose(self.z, other.z, 1e-9, 1e-9)


    def get_coord(self):
        return [self.x, self.y, self.z]
