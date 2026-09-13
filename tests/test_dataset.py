"""Scientific identity, selection and partition invariants."""

import numpy as np
import pandas as pd
import pytest

from psl_flexibility.dataset import (
    balanced_folds, cluster_sequences, nested_subcohorts, read_structure, sequence_similarity,
)


def atom(serial, number, *, chain="A", insertion=" ", alt=" ", occupancy=1.0,
         bfactor=10.0, x=None, name="ALA"):
    x = float(serial) if x is None else x
    return (f"ATOM  {serial:5d}  CA {alt}{name:3s} {chain}{number:4d}{insertion}   "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{bfactor:6.2f}           C\n")


def test_first_model_altloc_and_insertion_identity(tmp_path):
    path = tmp_path / "CASE.pdb"
    path.write_text("MODEL        1\n" + atom(1, 7, alt="A", occupancy=0.4, bfactor=9) +
                    atom(2, 7, alt="B", occupancy=0.6, bfactor=12) +
                    atom(3, 7, insertion="A", bfactor=18) + atom(4, 8, chain="B", bfactor=24) +
                    "ENDMDL\nMODEL        2\n" + atom(5, 99, bfactor=99) + "ENDMDL\n")
    frame, audit = read_structure(path)
    assert frame["residue_key"].tolist() == ["1|A|7|.", "1|A|7|A", "1|B|8|."]
    assert frame["altloc"].tolist() == ["B", ".", "."]
    assert frame["b_factor"].tolist() == [12, 18, 24]
    assert audit["discarded_altloc_ca"] == 1
    assert audit["excluded_other_model_ca"] == 1
    assert frame["z_b_factor"].mean() == pytest.approx(0)
    assert frame["z_b_factor"].std(ddof=0) == pytest.approx(1)


@pytest.mark.parametrize("kind", ["duplicate_key", "duplicate_coordinates", "constant_target"])
def test_reject_ambiguous_or_invalid_structure(tmp_path, kind):
    records = [atom(1, 1, bfactor=10), atom(2, 2, bfactor=20), atom(3, 3, bfactor=30)]
    if kind == "duplicate_key":
        records.append(atom(4, 1, bfactor=40))
    elif kind == "duplicate_coordinates":
        records[1] = atom(2, 2, bfactor=20, x=1)
    else:
        records = [atom(i, i, bfactor=10) for i in range(1, 4)]
    path = tmp_path / "invalid.pdb"
    path.write_text("".join(records))
    with pytest.raises(ValueError):
        read_structure(path)


def test_calcium_ions_not_protein_and_altloc_tie(tmp_path):
    path = tmp_path / "case.pdb"
    lines = [atom(1, 1, alt="A", occupancy=0.5, bfactor=10),
             atom(2, 1, alt=" ", occupancy=0.5, bfactor=15),
             atom(3, 2, bfactor=20), atom(4, 3, bfactor=30), atom(5, 4, name="CA", bfactor=40)]
    path.write_text("".join(lines))
    frame, audit = read_structure(path)
    assert len(frame) == 3 and frame.iloc[0]["atom_serial"] == 2
    assert audit["excluded_nonprotein_ca"] == 1


def test_similarity_uses_sequence_and_shorter_coverage():
    result = sequence_similarity("AAAAACCCCC", "CCCCC")
    assert result["identity"] == 1
    assert result["coverage_shorter"] == 1
    assert result["aligned_pairs"] == 5
    unknown = sequence_similarity("XXXXX", "XXXXX")
    assert unknown["identity"] == 0


def test_components_connect_any_chain_transitively():
    chains = {"A": {"A": "AAAAAAAAAA"}, "B": {"A": "AAAAAAAAAA", "B": "RRRRRRRRRR"},
              "C": {"A": "RRRRRRRRRR"}, "D": {"A": "CCCCCCCCCC"}}
    groups, edges = cluster_sequences(chains, minimum_aligned_pairs=0, reciprocal=False, exact_duplicates=False)
    assert groups["A"] == groups["B"] == groups["C"]
    assert groups["D"] != groups["A"]
    assert len(edges) == 2
    assert {"identity", "coverage_shorter", "chain_a", "chain_b"}.issubset(edges.columns)


def test_short_fragments_do_not_bridge_but_complete_duplicates_group():
    chains = {"A": {"A": "A" * 60, "B": "R"}, "B": {"A": "R" * 60},
              "C": {"A": "ACDEFG"}, "D": {"X": "ACDEFG"}}
    groups, edges = cluster_sequences(chains)
    assert groups["A"] != groups["B"]
    assert groups["C"] == groups["D"]
    assert len(edges) == 1 and edges.iloc[0]["edge_type"] == "exact_complete_record"


def test_folds_are_order_independent_and_keep_groups_intact():
    sizes = {f"P{i}": 10 + i for i in range(12)}
    groups = {p: "shared" if p in {"P0", "P1"} else p for p in sizes}
    first = balanced_folds(sizes, groups)
    reverse = balanced_folds(dict(reversed(list(sizes.items()))), dict(reversed(list(groups.items()))))
    assert first == reverse
    assert first["P0"] == first["P1"]
    assert set(first.values()) == set(range(1, 6))
    with pytest.raises(ValueError, match="independent groups"):
        balanced_folds(sizes, {p: "one" for p in sizes})


def test_subcohorts_nested_prespecified_and_case_included():
    ids = ["1ULR", "1X3O"] + [f"P{i:03}" for i in range(98)]
    frame = pd.DataFrame({"protein_id": ids, "n_residues": np.arange(100) + 80})
    cohort = nested_subcohorts(frame)
    a = set(cohort.loc[cohort["nested60"], "protein_id"])
    b = set(cohort.loc[cohort["nested20"], "protein_id"])
    assert len(a) == 60 and len(b) == 20
    assert {"1ULR", "1X3O"}.issubset(b) and b.issubset(a)
    reversed_cohort = nested_subcohorts(frame.iloc[::-1]).set_index("protein_id")
    pd.testing.assert_frame_equal(cohort.set_index("protein_id").sort_index(), reversed_cohort.sort_index())
