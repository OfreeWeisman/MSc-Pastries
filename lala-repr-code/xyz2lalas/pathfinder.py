import itertools

import networkx as nx

from .path import Path

def find(_knots):
    """
    find(_knots: list(Knot)) -> list(predixy.path.Path)

    Given a list of Knot objects, find path(s) through them and return them as a list of Path objects.

    in:
    _knots: A list of Knot objects.

    out:
    paths: A list of Path objects.

    """
    edges = get_connectivity(_knots)
    start_end_pairs = get_start_end(edges, len(_knots))
    graph = nx.Graph(edges) # generate mathematical graph as networkx Graph object
    paths = []
    for pair in start_end_pairs:
        for path in nx.all_shortest_paths(graph, pair[0], pair[1]): # generate all possible shortest paths through the connected Knot objects given a start and end Knot.
            paths.append(Path(*pair, [_knots[i] for i in path]))
    return paths, graph, edges


def get_connectivity(_knots):
    """
    get_connectivity(_knots: list(Knot)) -> list(tuple)

    Find out which Knot objects are connected and return those connections as a list of tuples ( = edges).

    in:
    _knots: A list of Knot objects.

    out:
    edges: A list of tuples ( = edges) which represent which Knot objects are connected.

    """
    edges = []
    for i in range(len(_knots)):
        for j in range(i + 1, len(_knots)):
            i_atoms = set(_knots[i].atoms)
            j_atoms = set(_knots[j].atoms)
            if i_atoms & j_atoms:
                edges.append((i,j))
    return edges


def get_start_end(_edges, _number_of_knots):
    """
    get_start_end(_edges: list(tuple), _number_of_knots: int) -> list(tuple)

    A function to find out all the Knot objects which are potential start/end points as they are only connected to one other Knot.

    in:
    _edges: A list of tuples. Each tuple gives the connectivitiy of one Knot to another.
    _number_of_knots: Total number of Knot objects.

    out:
    start_end_pairs: A list of tuples representing all possible unique combinations of start/end Knot objects.

    """

    counter_dict = {i : 0 for i in range(_number_of_knots)} # set up a dict with Knot index as key and 0 as the counter.
    for edge in _edges: # for each edge, add 1 to each counter for the relevant Knot.
        counter_dict[edge[0]] += 1
        counter_dict[edge[1]] += 1

    endpoint_list = [i for i in counter_dict.keys() if counter_dict[i] == 1] # generate a list of all endpoints, i.e. Knot objects with only one connection.

    start_end_pairs = [(x,y) for x, y in itertools.combinations(endpoint_list, 2)] # generate a list of tuples of all possible unique combinations
    start_end_pairs.extend([(y,x) for (x, y) in start_end_pairs])
    return start_end_pairs
