"""
Code to recursively go through network of knots (binary tree) and give out an array of Knot numbers.

It is a BINARY tree, so it can only branch once each time.

"""

import copy
import numpy as np
import pandas as pd


class Node():

    def __init__(self, _index):
        self.index = _index
        self.annulation = None
        self.children = []

    def count_children(self):
        return len(self.children)

    def __str__(self):
        return f"Node(index = {self.index}, annulation = {self.annulation}, #children = {len(self.children)})"


_LALAS_LETTER = {'0': 'L', '1': 'A', '2': 'a'}


def compute_canonical_sequence(_paths, _edges, _knots, _graph):
    """
    Run the LALAS canonicalisation pipeline.

    For every longest path through the knot graph, build a binary tree, compute
    annulations and the resulting raw sequence. Then expand each raw sequence
    with its inversion, deduplicate, and apply the Balaban-style filter to pick
    the canonical sequence.

    Returns
    -------
    final_seq : list
        The canonical sequence (digits 0/1/2 interspersed with '(' and ')').
    raw_sequences : list[list]
        One pre-inversion sequence per longest path (parallel to the trees /
        path indices below).
    raw_trees : list[Node]
        The root Node of the tree built for each longest path.
    raw_paths_idx : list[list[int]]
        The longest-path knot indices used to build each tree.
    """
    raw_sequences = []
    raw_trees = []
    raw_paths_idx = []

    for path in find_longest_path(_paths, _graph):
        longest_path = [knot.index for knot in path.route]
        root = Node(longest_path[0])
        edges_copy = copy.deepcopy(_edges)
        build_tree(root, edges_copy)
        get_annulation(root, None, _knots, longest_path)

        raw_sequences.append(get_sequence(longest_path, root))
        raw_trees.append(root)
        raw_paths_idx.append(longest_path)

    all_sequences = get_allsequences(raw_sequences)

    df = pd.DataFrame(all_sequences)
    df.drop_duplicates(inplace=True)
    no_duplicates = df.values.tolist()

    final_seq = filter_sequences(no_duplicates)
    return final_seq, raw_sequences, raw_trees, raw_paths_idx


def format_lalas_strings(final_seq):
    """
    Render the canonical sequence as the three string flavours used by the
    pipeline.

    Returns
    -------
    balaban_name : str
        The raw digits ('0', '1', '2') concatenated.
    augmented_lalas : str
        Digits replaced with 'L', 'A', 'a' (preserves the L/A/a distinction).
    lalas : str
        Same as augmented but with all 'a' folded back into 'A'.
    """
    balaban_name = "".join(map(str, final_seq))
    augmented_lalas = balaban_name
    for digit, letter in _LALAS_LETTER.items():
        augmented_lalas = augmented_lalas.replace(digit, letter)
    lalas = augmented_lalas.replace('a', 'A')
    return balaban_name, augmented_lalas, lalas


def get_LALAS(_paths, _edges, _knots, _graph):
    """Print 'balaban,augmented_lalas,lalas' for the canonical sequence."""
    final_seq, _, _, _ = compute_canonical_sequence(_paths, _edges, _knots, _graph)
    balaban_name, augmented_lalas, lalas = format_lalas_strings(final_seq)
    print(f'{balaban_name},{augmented_lalas},{lalas}')
    

def find_longest_path(_paths, _graph):
    # look for longest path(s)
    longest_paths = []
    longest_l = 0
    for path in _paths:
        l = len(path.route)
        if l > longest_l:
            longest_l = l
            longest_paths = [path]
        elif l == longest_l:
            longest_paths.append(path)

    # look for longest path with highest number of branches
    paths_branches = []
    if len(longest_paths) > 1:
        for path in longest_paths:
            nbranches = 0
            for knot in path.route:
                degree = _graph.degree[knot.index]
                if degree > 2:
                    nbranches += 1
            paths_branches.append(nbranches)

    max_branches_paths = []
    max_branches = max(paths_branches)
    for i in range(0, len(longest_paths)):
        if paths_branches[i] == max_branches:
            max_branches_paths.append(longest_paths[i])

    return max_branches_paths # return all longest paths with the same number of max branches (this also inclues the same path in both directions)


def build_tree(current_node, edges):
    # recursive tree building

    # breaking condition: we're out of edges
    if len(edges) < 1:
        return

    # run through edges and find all children
    for i in range(len(edges) -1, -1, -1):
        if current_node.index == edges[i][0]:
            current_node.children.append(Node(edges[i][1]))
            edges.pop(i)
        elif current_node.index == edges[i][1]:
            current_node.children.append(Node(edges[i][0]))
            edges.pop(i)

    # recursion
    for child in current_node.children:
        build_tree(child, edges)

    
def get_annulation(current_node, previous_node, _knots, _longest_path):
    if len(current_node.children) < 1:
        return # nothing to do, found an end

    elif len(current_node.children) > 1: # found a branch, must be "A"/1 or "a"/2
        if current_node.index in _longest_path:
            for child in current_node.children:
                if child.index in _longest_path:
                    if get_rotation(np.array(_knots[previous_node.index].get_coord()), # check which one
                                    np.array(_knots[current_node.index].get_coord()), 
                                    np.array(_knots[child.index].get_coord())) < 0:
                        current_node.annulation = 1
                    else: 
                        current_node.annulation = 2
                else: 
                    get_annulation(child, current_node, _knots, _longest_path)
        
        else:
            paths = []
            longest_path = 0
            for child in current_node.children:
                path = longestPath(child)
                paths.append(path)
                if len(path) > longest_path:
                    longest_path = len(path)

            new_main_path = [paths[i] for i, x in enumerate(paths) if len(x) == longest_path]

            # check if new main path is degenerate:
            if len(new_main_path) == 1:
                for child in current_node.children:
                    get_annulation(child, current_node, _knots, new_main_path)
            elif len(new_main_path) > 1 and longest_path == 0:
                current_node.annulation = 1
            else: print(f'Degenerate branching point with branches > 1! This case is not handled')

        for child in current_node.children:
            get_annulation(child, current_node, _knots, _longest_path)

    else:
        if previous_node: # prevent this from being called on the root node
            if get_angle(np.array(_knots[previous_node.index].get_coord()), 
                         np.array(_knots[current_node.index].get_coord()), 
                         np.array(_knots[current_node.children[0].index].get_coord())) < -0.87:
                current_node.annulation = 0 # "L"
            else:
                if get_rotation(np.array(_knots[previous_node.index].get_coord()), 
                                np.array(_knots[current_node.index].get_coord()), 
                                np.array(_knots[current_node.children[0].index].get_coord())) < 0:
                    current_node.annulation = 1
                else:
                    current_node.annulation = 2
        get_annulation(current_node.children[0], current_node, _knots, _longest_path)


def get_sequence(path, current_node):

    sequence = []
    # no branching 
    while current_node.count_children() == 1:
        if current_node.annulation is not None: 
            sequence.append(current_node.annulation)
        current_node = current_node.children[0]
    
    # branching
    # currently we do not check for longest branch in branches
    if current_node.count_children() > 1:
        if current_node.annulation: sequence.append(current_node.annulation)
        sequence.append("(")
        if current_node.children[1].index in path:
            sequence.extend(get_sequence(path, current_node.children[0]))
            sequence.append(")")
            sequence.extend(get_sequence(path, current_node.children[1]))
        else:
            sequence.extend(get_sequence(path, current_node.children[1]))
            sequence.append(")")
            sequence.extend(get_sequence(path, current_node.children[0]))

    # break condition
    if current_node.count_children() < 1:
        if current_node.annulation: sequence.append(current_node.annulation)
        return sequence
    
    return sequence


def get_allsequences(sequences):

    all_sequences = []
    for sequence in sequences:
        all_sequences.append(sequence)
        inverted_seq = []
        for i in range(0, len(sequence)):
            if isinstance(sequence[i], int):
                if sequence[i] == 0: 
                    inverted_seq.append(sequence[i])
                elif sequence[i] == 1: inverted_seq.append(2)
                else: inverted_seq.append(1)
            else: inverted_seq.append(sequence[i])
        all_sequences.append(inverted_seq)
    
    return all_sequences

def filter_sequences(_sequences):
    # Look for sequence with lowest number just main branch
    all_main = []
    for sequence in _sequences:
        w = 0
        stripped = []
        for s in sequence:
            if isinstance(s, int) and w == 0: stripped.append(s)
            elif s == '(': w += 1 
            elif s == ')': w -= 1
        all_main.append(int("".join(map(str,stripped))))
    
    min_stripped = min(all_main)
    filter_1 = [_sequences[i] for i, x in enumerate(all_main) if x == min_stripped]


    if len(filter_1) == 1:
        return filter_1[0]

    # Look for sequence with 1st branch the farthest away
    else:
        b_index = []
        for sequence in filter_1:
            b_index.append(sequence.index('('))
        
        max_b_id = max(b_index)
        filter_2 = [filter_1[i] for i, x in enumerate(b_index) if x == max_b_id]


        if len(filter_2) == 1:
            return filter_2[0]

        # get branching point positions and branch length
        else:
            positions = []
            llist = []
            for sequence in filter_2:
                w = 0
                close = []
                open = []
                for i, s in enumerate(sequence):
                    if isinstance(s, int): continue
                    elif s == '(':
                        if w == 0: open.append(i)
                        w += 1 
                    elif s == ')':
                        w -= 1
                        if w == 0: close.append(i)
                l = [y - x for x, y in zip(open, close)]
                llist.append(int("".join(map(str,l))))
                p = ""
                for i, x in enumerate(open):
                    if i == 0:
                        p += str(x)
                    else: 
                        p += str(abs(x - l[i-1]))
                positions.append(int(p))

            # select sequence with hightest position (branching farthest away)
            max_indices = max(positions)
            filter_3 = [filter_2[i] for i, x in enumerate(positions) if x == max_indices]


            if len(filter_3) == 1:
                return filter_3[0]

            # select sequence with longest branch farthest
            else:
                min_l_id = min(llist)
                filter_4 = [filter_3[i] for i, x in enumerate(llist) if x == min_l_id]


            if len(filter_4) == 1:
                return filter_4[0]

            # Look for sequence with lowest number with branches (remove brackets)
            else:
                all_stripped = []
                for sequence in filter_4:
                    stripped = []
                    for s in sequence:
                        if isinstance(s, int):
                            stripped.append(s)
                    all_stripped.append(int("".join(map(str,stripped))))
                
                min_stripped = min(all_stripped)
                filter_5 = [filter_4[i] for i, x in enumerate(all_stripped) if x == min_stripped]


                if len(filter_5) == 1:
                    return filter_5[0]
                else:
                    return min(filter_5, key=lambda seq: "".join(map(str, seq)))

def longestPath(root):

    # If No. children < 1 means there is no binary tree so eturn a empty vector
    if root.count_children() < 1:
        return []
 
    elif root.count_children() == 1:
        rightvect = longestPath(root.children[0])
        leftvect = []
 
    else:
        rightvect = longestPath(root.children[0])
        leftvect = longestPath(root.children[1])
    
    # Compare the size of the two vectors and insert current node accordingly
    if (len(leftvect) > len(rightvect)):
        leftvect.append(root.index)
    else:
        rightvect.append(root.index)
 
    # Return the appropriate vector
    if len(leftvect) > len(rightvect):
        return leftvect
 
    return rightvect


def print_tree(root):

    def rec_print(node):
        print(node)
        for child in node.children:
            rec_print(child)

    rec_print(root)
    

def print_longestPath(root):
    def rec_print(node):
        print(node)
        print(longestPath(node))
        for child in node.children:
            rec_print(child)
    rec_print(root)


def get_angle(a, b, c):

    ba = a - b
    bc = c - b

    return np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

def get_rotation(a, b, c):

    ba = a - b
    bc = c - b

    return int(np.sign(np.cross(ba,bc)[2]))
