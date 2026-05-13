from dataclasses import dataclass

__all__ = ['Knot']


@dataclass
class Knot:
    index: int
    cycle_type: str
    atoms: list
    x: float
    y: float
    z: float

    def __repr__(self):
        atoms = "\n  ".join([str(atom) for atom in self.atoms])
        return f"Knot(\n  index: {self.index}\n  cycle_type: {self.cycle_type}\n  center: {self.x:8.5f} {self.y:8.5f} {self.z:8.5f}\n  atoms:\n  {atoms}\n  )"

    def __str__(self):
        atoms = "\n  ".join([str(atom) for atom in self.atoms])
        return f"  index: {self.index}\n  cycle_type: {self.cycle_type}\n  center: {self.x:8.5f} {self.y:8.5f} {self.z:8.5f}\n  atoms:\n  {atoms}"

    def get_opposing_edge(self, _edge):
        """
        Returns the opposing edge or atom (for 5-membered ring case B) of the inputed edge.

        """

        opposite_atoms = []

        n_atoms = len(self.atoms)
        n_half = n_atoms // 2

        if n_atoms == 5:
            if self.cycle_type in ['furb', 'pylb', 'thib']:
                for atom in self.atoms:
                    if atom.element in ['O', 'N', 'S']:
                        return (atom,)
            
            else:
                for atom in self.atoms:
                    if not (atom.index == _edge[0].index or atom.index == _edge[1].index or atom.element in ['N', 'O', 'S']):
                        opposite_atoms.append(atom)
                return (atom for atom in opposite_atoms)

        else:
            a = self.atoms.index(_edge[0])
            b = self.atoms.index(_edge[1])

            if max(a,b) == n_atoms - 1:
                if min(a,b) == 0:
                    start_index = 0
                else:
                    start_index = max(a,b)
            else:
                start_index = max(a,b)
            
            opposite_atoms.append((start_index + n_half) % n_atoms)
            opposite_atoms.append((start_index + n_half - 1) % n_atoms)
            return (self.atoms[atom] for atom in opposite_atoms)
        


    def get_coord(self):
        return [self.x, self.y, self.z]