# Text-Based Representations with Interpretable Machine Learning Reveal Structure-Property Relationships for Polybenzenoid Hydrocarbons

## Content
* Source code for calculations
* Source code for obtaining the LALAS and augmented-LALAS representations
* Source code for obtaining the LALAS/augmented-LALAS feature vectors (LFVs)
* Data used as input (COMPAS-1D dataset and generated LFVs)

## Generating representations for COMPAS-2x (PAS)

`lala-repr-code/scripts/sdf2lalas_csv.py` generates the augmented-LALAS representation for
every charge-0 molecule of a COMPAS SDF and merges it with the COMPAS feature table:

```bash
cd lala-repr-code
python scripts/sdf2lalas_csv.py --sdf ../data/compas-2x.sdf \
                                --features-csv ../data/compas-2x.csv \
                                --out-dir ../data/results
```

Outputs (`data/results/`): `compas-2x_lalas.csv` (`name,representation`) and
`compas-2x_lalas_features.csv` (the same representations joined to the COMPAS features).
The SDF holds one record per charge state; only `charge == 0` is used. Molecules with
fewer than three rings are skipped (74 in COMPAS-2x), leaving 524,318 representations.

### Representation format

```
<ring types>  <augmented LALAS>
ben,ben,(pyd[N:1]),fur[O:3]  LA()a
```

The ring-type part lists the rings along the canonical traversal, with branches in
parentheses; the LALAS part is the canonical annulation string (`L` linear, `A`/`a`
angular). Ring types: `ben` benzene, `pyd` pyridine, `pyz` pyrazine, `brn` borinine,
`dbrn` 1,4-diborinine, `dhdb` dihydro-diborinine, `bor` borole, `pyl` pyrrole,
`fur` furan, `thi` thiophene, `cbd` cyclobutadiene. A heteroatom-bearing ring carries a
token such as `[N:1]` or `[B:1,4]`: each ring is numbered `0..n-1` from a fusion atom
with its parent ring, so the positions are relative to the traversal, not to IUPAC
numbering.

### Canonicalisation rules

1. **Rings and ring types.** Rings come from the SDF bond block (`nx.minimum_cycle_basis`);
   a ring type follows from its size and heteroatom content. The one ambiguous case is the
   six-ring with two borons: `dhdb` when each boron carries an explicit B-H bond, otherwise
   `dbrn`. This uses the bond table, not interatomic distances.
2. **Smallest LALAS wins.** The canonical annulation string is chosen first, over all
   longest paths, both traversal directions and their inversions. Nothing below can
   override it.
3. **Smallest representation among equal LALAS.** Several traversals may read as the same
   LALAS; the one whose rendered ring-type string (heteroatom positions included) is
   lexicographically smallest, token by token, is kept.
4. **Numbering direction.** Heteroatom positions are numbered in one direction for the
   whole molecule, taken from the first angular annulation. When the molecule has none
   (all-`L`) or is symmetric (`ambiguous`), geometry does not fix a direction, so both are
   tried and the smaller token sequence is kept.
5. **Forks of end rings.** At a branch point carrying two end rings, both orderings are
   generated as candidate traversals and their annulation is computed from the geometry,
   so rule 2 picks the layout. Because the two end rings lie on opposite sides, the
   canonical string always shows `A` there, and the ring order inside the branch follows
   the geometry rather than the lexicographic order of the ring types.

Rules 2-5 make the output independent of the molecule's orientation: a molecule and its
mirror image give the same representation. This matters in practice, since the alignment
step (`Mol.align_to_xy_plane`) can produce either frame on different machines.

### How this was verified

Against the COMPAS feature table (ground truth) for all 524,318 molecules:

* total ring count equals the `rings` column for every molecule, and no ring falls outside
  the 11 known types;
* all 11 per-ring-type columns match for 516,630 molecules. The remaining 7,688 differ only
  in `benzene`, where the feature table contradicts itself: its per-type columns sum to one
  more than its own `rings` total. Our count matches `rings`;
* the 1,011 molecules previously known to report a dhdb ring as dbrn all match after rule 1.

Invariance and stability:

* mirroring a molecule after alignment leaves the representation unchanged - checked on
  ~10.5K molecules, on all fork molecules of a 78K run, and on a 2K control sample;
* rerunning the pre-change code (commits `a79b9fa`, `757616e`) reproduces the previous
  LALAS exactly on 21,733 molecules, so the only LALAS changes introduced are the 322 fork
  molecules where `a` became `A` (796 molecules contain such a fork). No other LALAS in the
  dataset changed, and no change at a non-fork position occurred.

`data/results/manual_check_sample.csv` holds 10 randomly selected molecules per category
(each fix plus unchanged molecules) with formula and SMILES, for manual spot checks.

Both automated checks are kept as a regression test:

```bash
python verification/verify_representation.py --sample 2000
```

It regenerates a random sample of molecules, requires each one to be unchanged when
mirrored, and compares its ring-type counts against the feature table (skipping the
self-contradicting rows described above). See the script's header for details.

## How to cite this work
If you use any part of this work, please cite the following:
S. Fite, A. Wahab, E. Paenurk, Z. Gross, and R. Gershoni-Poranne, Text-based representations with interpretable machine learning reveal structure–property relationships of polybenzenoid hydrocarbons, DOI: 10.1002/poc.4458

## Support
For support or to report any issues, please contact: porannegroup /at/ technion.ac.il

## Authors and acknowledgment
This work was conducted under the supervision of Prof. Dr. Renana Gershoni-Poranne (Technion/ETH Zurich). The following people contributed to this project: 
1. Shachar Fite (Technion)
2. Alexandra Wahab (ETH Zurich)
3. Dr. Eno Paenurk (ETH Zurich)

The invaluable assistance of the following people is gratefully acknowledged: Prof. Dr. Peter Chen, Prof. Zeev Gross, Dr. Alexandra Tsybizova, Felix Fleckenstein. 

In addition, the financial support of the Branco Weiss Fellowship is acknowledged.

## License
The code and data in this repository are provided free-of-charge. They are licensed under a CC-BY-NC-SA license. Please cite the relevant literature if you use them.
