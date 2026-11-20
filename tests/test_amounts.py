#!/usr/bin/env python
import logging
import os
import unittest
import unittest.mock
from decimal import Decimal

import pytest

import pdfplumber
from pdfplumber import amounts
from pdfplumber.amounts import (
    CellInput,
    CellState,
    ConstraintStatus,
    NumberFormat,
    Relation,
    SumConstraint,
    parse_candidates,
)

logging.disable(logging.ERROR)

HERE = os.path.abspath(os.path.dirname(__file__))

EN = NumberFormat.en()
DE = NumberFormat.de()
PLAIN = NumberFormat.plain()
D = Decimal


class TestNumberFormat(unittest.TestCase):
    def test_en_parse(self):
        assert EN.parse("1,000") == D("1000")
        assert EN.parse("1,000.50") == D("1000.50")
        assert EN.parse("1000") == D("1000")
        assert EN.parse("0.61") == D("0.61")
        assert EN.parse("1.000,00") is None
        assert EN.parse("1,00") is None  # broken thousands group
        assert EN.parse("1,0000") is None
        assert EN.parse("1..0") is None
        assert EN.parse("abc") is None
        assert EN.parse("") is None
        assert EN.parse("1 000") is None  # space not part of this format

    def test_de_parse(self):
        assert DE.parse("1.000") == D("1000")
        assert DE.parse("1,000") == D("1.000")
        assert DE.parse("1.000,50") == D("1000.50")
        assert DE.parse("1,000.50") is None

    def test_fr_parse(self):
        fr = NumberFormat.fr()
        assert fr.parse("1 000,00") == D("1000.00")
        assert fr.parse("1\u00a0000") == D("1000")
        assert fr.parse("1\u202f000") == D("1000")

    def test_plain_parse(self):
        assert PLAIN.parse("1000.5") == D("1000.5")
        assert PLAIN.parse("1,000") is None

    def test_ch_parse(self):
        assert NumberFormat.ch().parse("1'000'000.00") == D("1000000.00")

    def test_signs(self):
        assert EN.parse("-1,000") == D("-1000")
        assert EN.parse("\u22121,000") == D("-1000")  # U+2212 minus
        assert EN.parse("1,000-") == D("-1000")
        assert EN.parse("(1,000.00)") == D("-1000.00")
        assert EN.parse("+1,000") == D("1000")
        assert EN.parse("( 1,000 )") == D("-1000")

    def test_symbols(self):
        fmt = NumberFormat.en(symbols=("$",))
        assert fmt.parse("$ 1,000.00") == D("1000.00")
        assert fmt.parse("1,000.00 $") == D("1000.00")
        assert fmt.parse("-$ 1,000") == D("-1000")
        assert fmt.parse("($ 1,000)") == D("-1000")
        de_eur = NumberFormat.de(symbols=("€",))
        assert de_eur.parse("1.000,00 €") == D("1000.00")
        assert de_eur.parse("€ 1.000,00") == D("1000.00")

    def test_exactness(self):
        value = EN.parse("0.1")
        assert value == D("0.1")
        assert isinstance(value, Decimal)
        assert EN.parse("1,000.005") == D("1000.005")
        # beyond float precision
        assert PLAIN.parse("123456789012345678901234567890") == D(
            "123456789012345678901234567890"
        )

    def test_format_validation(self):
        with pytest.raises(ValueError):
            NumberFormat(thousands="", decimal="")
        with pytest.raises(ValueError):
            NumberFormat(thousands=",", decimal=",")
        with pytest.raises(ValueError):
            NumberFormat(thousands="1", decimal=".")
        with pytest.raises(ValueError):
            NumberFormat(thousands=",", decimal=".", symbols=("",))

    def test_parse_candidates(self):
        assert parse_candidates("1,000", [EN, DE]) == (D("1.000"), D("1000"))
        assert parse_candidates("1000", [EN, DE]) == (D("1000"),)
        assert parse_candidates("abc", [EN, DE]) == ()
        assert parse_candidates("1,000", [EN]) == (D("1000"),)


class TestClassification(unittest.TestCase):
    def test_states(self):
        result = amounts.clean(
            [["1,000", "", None, "Revenue", "1,00,000"]],
            formats=[EN, DE],
        )
        cells = result.cells[0]
        assert cells[0].state is CellState.VALUE
        assert cells[0].possible == (D("1.000"), D("1000"))
        assert cells[1].state is CellState.BLANK
        assert cells[1].value is None
        assert cells[2].state is CellState.MISSING
        assert cells[3].state is CellState.TEXT
        assert cells[4].state is CellState.UNPARSEABLE
        # none of the non-numeric cells may produce a value
        for cell in cells[1:]:
            assert cell.value is None
            assert cell.possible == ()

    def test_unconstrained_cells(self):
        result = amounts.clean([["1,000", "1000"]], formats=[EN, DE])
        ambiguous, plain = result.cells[0]
        # no relations: every candidate remains possible
        assert ambiguous.value is None
        assert ambiguous.ambiguous
        assert ambiguous.possible == (D("1.000"), D("1000"))
        # a single candidate is trivially unique
        assert plain.value == D("1000")
        assert plain.resolved

    def test_text_and_bbox_preserved(self):
        bbox = (1, 2, 3, 4)
        result = amounts.clean(
            [[("1,000", bbox), CellInput(text="x", bbox=None)]],
            formats=[EN, DE],
        )
        assert result.cells[0][0].text == "1,000"
        assert result.cells[0][0].bbox == bbox
        assert result.cells[0][1].text == "x"

    def test_ragged_grid_padded_with_missing(self):
        result = amounts.clean([["1", "2"], ["3"]], formats=[PLAIN])
        assert len(result.cells[1]) == 2
        assert result.cells[1][1].state is CellState.MISSING

    def test_bad_cell_type(self):
        with pytest.raises(TypeError):
            amounts.clean([[object()]], formats=[EN])


class TestSolving(unittest.TestCase):
    def test_row_and_column_subtotals_disambiguate(self):
        grid = [
            ["Item", "Q1", "Q2", "Total"],
            ["A", "1,000", "2,500", "3,500"],
            ["B", "500", "1,000", "1,500"],
            ["Total", "1,500", "3,500", "5,000"],
        ]
        result = amounts.clean(
            grid,
            formats=[EN, DE],
            relations=[
                Relation(axis="row", total=3, addends=(1, 2), within=(1, 2, 3)),
                Relation(axis="column", total=3, addends=(1, 2), within=(1, 2, 3)),
            ],
        )
        assert result.complete
        assert result.conflicts == ()
        expected = {
            (1, 1): "1000",
            (1, 2): "2500",
            (1, 3): "3500",
            (2, 1): "500",
            (2, 2): "1000",
            (2, 3): "1500",
            (3, 1): "1500",
            (3, 2): "3500",
            (3, 3): "5000",
        }
        for (r, c), value in expected.items():
            cell = result.cell(r, c)
            assert cell.value == D(value), (r, c, cell)
            assert cell.constrained
            # solved values keep the exact parsed candidate form
            assert cell.value in cell.candidates
        assert result.cell(0, 0).state is CellState.TEXT

    def test_ambiguity_preserved_under_constraints(self):
        # (1,1,2) and (1000,1000,2000) both satisfy the total
        result = amounts.clean(
            [["1,000", "1,000", "2,000"]],
            formats=[EN, DE],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        x, y, z = result.cells[0]
        assert x.possible == (D("1.000"), D("1000"))
        assert y.possible == (D("1.000"), D("1000"))
        assert z.possible == (D("2.000"), D("2000"))
        assert x.value is None and y.value is None and z.value is None
        assert result.conflicts == ()

    def test_unique_value_despite_multiple_interpretations(self):
        # two global interpretations exist, but "0" is 0 in all of them
        result = amounts.clean(
            [["1,000", "1,000", "0", "2,000"]],
            formats=[EN, DE],
            relations=[Relation(axis="column", total=3, addends=(0, 1, 2))],
        )
        assert result.cell(0, 2).value == D("0")
        assert result.cell(0, 0).ambiguous
        assert result.complete

    def test_constraint_forces_interpretation(self):
        # "1,5" and "0,5" only parse under DE, forcing "1,000" -> 1
        result = amounts.clean(
            [["1,000", "0,5", "1,5"]],
            formats=[EN, DE],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert result.cell(0, 0).value == D("1.000")
        assert result.cell(0, 1).value == D("0.5")
        assert result.cell(0, 2).value == D("1.5")

    def test_conflict_reported(self):
        result = amounts.clean(
            [["1,000", "1,000", "3,5"]],
            formats=[EN, DE],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        conflicts = result.conflicts
        assert len(conflicts) == 1
        report = conflicts[0]
        assert report.status is ConstraintStatus.CONFLICT
        assert report.total == (0, 2)
        assert report.satisfiable_alone is False
        for cell in result.cells[0]:
            assert cell.value is None
            assert cell.possible == ()
            assert cell.conflicted
        assert result.to_rows() == [[None, None, None]]

    def test_blank_is_not_zero(self):
        result = amounts.clean(
            [["10", "", "10"]],
            formats=[PLAIN],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert result.conflicts == ()
        indeterminate = result.indeterminate
        assert len(indeterminate) == 1
        assert indeterminate[0].status is ConstraintStatus.INDETERMINATE
        blank = result.cell(0, 1)
        assert blank.state is CellState.BLANK
        assert blank.value is None  # never silently zero
        # the indeterminate constraint must not constrain the other cells
        assert result.cell(0, 0).value == D("10")
        assert result.cell(0, 2).value == D("10")

    def test_missing_is_not_zero(self):
        result = amounts.clean(
            [["10", None, "10"]],
            formats=[PLAIN],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert len(result.indeterminate) == 1
        assert result.cell(0, 1).state is CellState.MISSING
        assert result.cell(0, 1).value is None

    def test_unparseable_in_constraint_is_invalid(self):
        result = amounts.clean(
            [["10", "1,00,000", "10"]],
            formats=[EN],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert len(result.conflicts) == 1
        assert result.conflicts[0].status is ConstraintStatus.INVALID

    def test_no_float_approximation(self):
        # 0.1 + 0.2 is exactly 0.3; a float/epsilon implementation would
        # wrongly accept 0.30000000000000004 as the total
        result = amounts.clean(
            [["0.1", "0.2", "0.30000000000000004"]],
            formats=[PLAIN],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert len(result.conflicts) == 1
        ok = amounts.clean(
            [["0.1", "0.2", "0.3"]],
            formats=[PLAIN],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert ok.conflicts == ()
        assert ok.cell(0, 2).value == D("0.3")

    def test_big_integers_exact(self):
        big = "123456789012345678901234567890"
        result = amounts.clean(
            [[big, "1", "123456789012345678901234567891"]],
            formats=[PLAIN],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert result.conflicts == ()
        assert result.cell(0, 0).value == D(big)
        assert result.cell(0, 2).value == D("123456789012345678901234567891")

    def test_chained_subtotals(self):
        grid = [["1,000"], ["500"], ["1,500"], ["2,500"], ["4,000"]]
        result = amounts.clean(
            grid,
            formats=[EN, DE],
            relations=[
                Relation(axis="row", total=2, addends=(0, 1)),
                Relation(axis="row", total=4, addends=(2, 3)),
            ],
        )
        assert result.conflicts == ()
        values = [row[0].value for row in result.cells]
        assert values == [D("1000"), D("500"), D("1500"), D("2500"), D("4000")]

    def test_within_restricts_columns(self):
        grid = [
            ["1,000", "1,900"],
            ["500", "1,900"],
            ["1,500", "1,900"],
        ]
        result = amounts.clean(
            grid,
            formats=[EN, DE],
            relations=[Relation(axis="row", total=2, addends=(0, 1), within=(0,))],
        )
        # DE fails in column 0 (1 + 500 != 1.5), so EN is enforced there
        assert result.cell(0, 0).value == D("1000")
        # column 1 is outside the relation: ambiguity remains
        assert result.cell(0, 1).possible == (D("1.900"), D("1900"))

    def test_sum_constraint_explicit_cells(self):
        result = amounts.clean(
            [["1,000", "500"], ["1,500", ""]],
            formats=[EN, DE],
            relations=[SumConstraint(total=(1, 0), addends=((0, 0), (0, 1)))],
        )
        assert result.cell(1, 0).value == D("1500")
        assert result.cell(0, 0).value == D("1000")
        assert result.cell(0, 1).value == D("500")

    def test_budget_exhaustion_is_conservative(self):
        result = amounts.clean(
            [["1,000", "1,000", "2,000"]],
            formats=[EN, DE],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
            node_budget=1,
        )
        assert not result.complete
        assert any(r.status is ConstraintStatus.UNDECIDED for r in result.reports)
        # nothing may be resolved when the search could not finish
        for cell in result.cells[0]:
            assert cell.value is None
            assert cell.possible == cell.candidates

    def test_relation_validation(self):
        with pytest.raises(ValueError):
            Relation(axis="diagonal", total=1, addends=(0,))
        with pytest.raises(ValueError):
            Relation(axis="row", total=1, addends=(1,))
        with pytest.raises(ValueError):
            Relation(axis="row", total=2, addends=(0, 0))
        with pytest.raises(ValueError):
            Relation(axis="row", total=-1, addends=(0,))
        with pytest.raises(ValueError):
            SumConstraint(total=(0, 0), addends=((0, 0),))

    def test_relation_out_of_bounds(self):
        with pytest.raises(ValueError):
            amounts.clean(
                [["1", "2"]],
                formats=[PLAIN],
                relations=[Relation(axis="row", total=5, addends=(0,))],
            )
        with pytest.raises(ValueError):
            amounts.clean(
                [["1", "2"]],
                formats=[PLAIN],
                relations=[SumConstraint(total=(0, 9), addends=((0, 0),))],
            )

    def test_usage_errors(self):
        with pytest.raises(ValueError):
            amounts.clean([["1"]], formats=[])
        with pytest.raises(ValueError):
            amounts.clean([["1"]], formats=[EN], solution_limit=0)
        with pytest.raises(TypeError):
            amounts.clean([["1"]], formats=[EN], relations=["nope"])

    def test_to_rows_only_unique_values(self):
        result = amounts.clean(
            [["1,000", "1000"]],
            formats=[EN, DE],
        )
        assert result.to_rows() == [[None, D("1000")]]

    def test_mixed_formats_in_one_table(self):
        # "1,000" must be read as EN (1000) while "2,5" must be read as
        # DE (2.5) for the total "1,002.5" (EN-only) to hold
        result = amounts.clean(
            [["1,000", "2,5", "1,002.5"]],
            formats=[EN, DE],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert result.conflicts == ()
        assert result.cell(0, 0).value == D("1000")
        assert result.cell(0, 1).value == D("2.5")
        assert result.cell(0, 2).value == D("1002.5")

    def test_solution_limit_triggers_candidate_rechecks(self):
        # with solution_limit=1 the search stops early; the remaining
        # candidates are then checked individually and stay exact
        result = amounts.clean(
            [["1,000", "1,000", "2,000"]],
            formats=[EN, DE],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
            solution_limit=1,
        )
        assert result.complete
        assert result.cell(0, 0).possible == (D("1.000"), D("1000"))
        assert result.cell(0, 2).possible == (D("2.000"), D("2000"))

    def test_satisfiable_alone_budget_exhaustion(self):
        grid = [["1,000"] * 17 + ["3,5"]]
        relations = [Relation(axis="column", total=17, addends=tuple(range(17)))]
        with unittest.mock.patch.object(amounts, "_DIAGNOSTIC_BUDGET", 1):
            result = amounts.clean(grid, formats=[EN, DE], relations=relations)
        conflicts = result.conflicts
        assert len(conflicts) == 1
        assert conflicts[0].status is ConstraintStatus.CONFLICT
        assert conflicts[0].satisfiable_alone is None

    def test_search_seed_contradiction(self):
        domains = {(0, 0): (D("1"),), (0, 1): (D("2"),)}
        constraint = amounts._NormConstraint(
            total=(0, 1), addends=((0, 0),), label="", origin=None
        )
        solutions, exhausted = amounts._search(
            domains, [constraint], 100, 10, seed={(0, 0): D("1"), (0, 1): D("2")}
        )
        assert solutions == []
        assert exhausted

    def test_result_convenience_accessors(self):
        result = amounts.clean(
            [["1,000", "1000"]],
            formats=[EN, DE],
        )
        assert result.ambiguous_cells == (result.cell(0, 0),)
        assert result.resolved_cells == (result.cell(0, 1),)
        assert "resolved=1" in repr(result)
        assert "conflicts=0" in repr(result)

    def test_relation_labels(self):
        result = amounts.clean(
            [["1", "2", "3"]],
            formats=[PLAIN],
            relations=[Relation(axis="column", total=2, addends=(0, 1), label="check")],
        )
        assert result.reports[0].label == "check (row 0)"
        assert result.reports[0].status is ConstraintStatus.ENFORCED
        row_result = amounts.clean(
            [["1"], ["2"], ["3"]],
            formats=[PLAIN],
            relations=[Relation(axis="row", total=2, addends=(0, 1), label="check")],
        )
        assert row_result.reports[0].label == "check (column 0)"

    def test_negative_amounts_sum_exactly(self):
        result = amounts.clean(
            [["(100)", "(200)", "(300)"]],
            formats=[EN],
            relations=[Relation(axis="column", total=2, addends=(0, 1))],
        )
        assert result.conflicts == ()
        assert result.cell(0, 0).value == D("-100")
        assert result.cell(0, 2).value == D("-300")

    def test_recheck_budget_exhaustion_is_conservative(self):
        # the first interpretation is found within budget, but the
        # per-candidate re-checks exceed it: values stay possible
        result = amounts.clean(
            [["1,000", "1,000", "2,000", "3,000", "5,000"]],
            formats=[EN, DE],
            relations=[
                SumConstraint(total=(0, 2), addends=((0, 0), (0, 1))),
                SumConstraint(total=(0, 4), addends=((0, 2), (0, 3))),
            ],
            solution_limit=1,
            node_budget=3,
        )
        assert not result.complete
        assert result.cell(0, 0).possible == (D("1.000"), D("1000"))
        assert result.cell(0, 0).value is None

    def test_more_validation(self):
        with pytest.raises(ValueError):
            NumberFormat(thousands="..", decimal=".")
        with pytest.raises(ValueError):
            NumberFormat(thousands=",", decimal=".", symbols=("1$",))
        with pytest.raises(ValueError):
            Relation(axis="row", total=1, addends=())
        with pytest.raises(ValueError):
            SumConstraint(total=(0, 0), addends=())
        with pytest.raises(ValueError):
            SumConstraint(total=(0, 0), addends=((0, 1), (0, 1)))
        with pytest.raises(ValueError):
            SumConstraint(total=(0, 0), addends=((0, 1, 2),))
        with pytest.raises(ValueError):
            SumConstraint(total=(0, 0), addends=((0, -1),))
        with pytest.raises(ValueError):
            amounts.clean([["1"]], formats=[EN], node_budget=0)
        with pytest.raises(TypeError):
            amounts.clean([[(1, 2)]], formats=[EN])
        with pytest.raises(ValueError):
            amounts.clean([[("1", (1, 2, 3))]], formats=[EN])

    def test_parse_edge_cases(self):
        fmt = NumberFormat.en(symbols=("$",))
        assert fmt.parse("$") is None
        assert EN.parse("1 0") is None
        assert EN.parse(",") is None
        assert EN.parse("1,") is None
        assert EN.parse("()") is None


class TestTableIntegration(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        path = os.path.join(HERE, "pdfs/issue-140-example.pdf")
        cls.pdf = pdfplumber.open(path)

    @classmethod
    def teardown_class(cls):
        cls.pdf.close()

    def test_clean_table_preserves_text_and_positions(self):
        page = self.pdf.pages[0]
        table = max(page.find_tables(), key=lambda t: len(t.cells))
        fmt = NumberFormat.en(symbols=("$",))
        result = amounts.clean(table, formats=[fmt])

        assert len(result.cells) == len(table.rows)
        assert len(result.cells[0]) == len(table.rows[0].cells)

        header = result.cell(0, 0)
        assert header.state is CellState.TEXT
        assert header.text == "Line no"

        blank = result.cell(1, 0)
        assert blank.state is CellState.BLANK
        assert blank.value is None

        quantity = result.cell(1, 4)
        assert quantity.text == "637"
        assert quantity.value == D("637")
        assert quantity.bbox == table.rows[1].cells[4]

        amount = result.cell(1, 5)
        assert amount.text == "$ 0.61"
        assert amount.value == D("0.61")

        rate = result.cell(1, 7)
        assert rate.text == "0.0000"
        assert rate.value == D("0.0000")

    def test_ambiguous_amounts_in_real_table(self):
        page = self.pdf.pages[0]
        table = max(page.find_tables(), key=lambda t: len(t.cells))
        result = amounts.clean(table, formats=[EN, DE])
        # "637" parses identically under both formats -> resolved
        assert result.cell(1, 4).value == D("637")
        # "$ 0.61" matches no declared format -> unparseable, not zero
        assert result.cell(1, 5).state is CellState.UNPARSEABLE
        assert result.cell(1, 5).value is None
