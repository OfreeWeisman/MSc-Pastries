from dataclasses import dataclass


@dataclass
class Path:
    start: int
    end: int
    route : list # list of Knots

    def __repr__(self):
        knots = "\n  ".join(str(knot) for knot in self.route)
        return f"Path(\n  start: {self.start}\n  end: {self.end} \n  knots:\n  {knots}\n)"


    def __str__(self):
        knotlist = "\n  ".join(f"{str(knot.index)} : {str(knot.cycle_type)}" for knot in self.route)
        return f"  {knotlist}"