"""
Draw the heavy-atom bond graph of an xyz file using the same geometric
connectivity (+ valence pruning) as ``knotidentifier.identify`` when
``Mol.bonds`` is absent.

Saves a PNG next to the xyz (or to --out) for manual inspection.

Usage (from ``lala-repr-code/``):

    python scripts/plot_xyz_bond_graph.py PATH/TO/file.xyz [--out out.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LALA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LALA_DIR))

import matplotlib.pyplot as plt
import networkx as nx

from xyz2lalas import molloader, knotidentifier  # noqa: E402

COVALENCY = 1.3


def plot_xyz_bond_graph(xyz_path: Path, out_path: Path, *, show_hydrogen: bool = False) -> None:
    mol = molloader.load(str(xyz_path))
    _knots, edges = knotidentifier.identify(mol, COVALENCY)

    G = nx.Graph()
    for i, atom in enumerate(mol.atoms):
        G.add_node(i, element=atom.element, x=atom.x, y=atom.y)

    for u, v in edges:
        eu = mol.atoms[u].element
        ev = mol.atoms[v].element
        if not show_hydrogen and (eu == "H" or ev == "H"):
            continue
        G.add_edge(u, v)

    if not show_hydrogen:
        nodes_h = [n for n in G.nodes() if mol.atoms[n].element == "H"]
        G.remove_nodes_from(nodes_h)

    pos = {n: (mol.atoms[n].x, mol.atoms[n].y) for n in G.nodes()}

    fig, ax = plt.subplots(1, 1, figsize=(14, 12), dpi=150)
    ax.set_aspect("equal", adjustable="box")

    elist = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=elist, ax=ax, width=0.8, alpha=0.55, edge_color="#444")

    c_nodes = [n for n in G.nodes() if mol.atoms[n].element == "C"]
    other = [n for n in G.nodes() if mol.atoms[n].element != "C"]

    nx.draw_networkx_nodes(G, pos, nodelist=c_nodes, ax=ax, node_color="#5a5a5a",
                           node_size=80, linewidths=0.3, edgecolors="white")
    if other:
        nx.draw_networkx_nodes(G, pos, nodelist=other, ax=ax, node_color="#d62728",
                               node_size=120, linewidths=0.3, edgecolors="white")

    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=5, font_color="black")

    ax.set_title(f"{xyz_path.name}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\n"
                 f"(heavy only; same bonds as LALAS knotidentifier)", fontsize=10)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.margins(0.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"# saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path, help="Input .xyz file")
    parser.add_argument("--out", type=Path, default=None, help="Output .png path")
    parser.add_argument("--with-h", action="store_true", help="Include H in graph (cluttered)")
    args = parser.parse_args()

    xyz = args.xyz.expanduser().resolve()
    out = args.out or xyz.with_suffix(".png")
    plot_xyz_bond_graph(xyz, out, show_hydrogen=args.with_h)


if __name__ == "__main__":
    main()
