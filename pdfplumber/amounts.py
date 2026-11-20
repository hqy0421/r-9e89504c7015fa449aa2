"""
Exact, constraint-based cleaning of locale-ambiguous amounts in tables.

pdfplumber extracts the *text* of each table cell exactly as it appears in
the PDF. In financial tables, that text often mixes regional number formats:
``1,000`` may mean one thousand (thousands separator ``,``) or exactly one
(decimal separator ``,``). This module provides :func:`clean`, an entry
point that resolves such ambiguity *globally* instead of guessing per cell:

1. Each cell's raw text is parsed under every declared :class:`NumberFormat`,
   producing a small set of exact :class:`~decimal.Decimal` candidates.
2. Known row/column subtotal relations are expanded into exact sum
   constraints over those candidates.
3. A constraint solver searches for *all* table-wide interpretations that
   satisfy every total simultaneously.
4. A cell's cleaned value is kept only when it is identical across **all**
   feasible interpretations. Cells that remain ambiguous report the values
   still possible. Totals that cannot be reconciled are reported as
   conflicts.

Guarantees:

* All arithmetic is exact decimal/integer arithmetic. Floats are never
  used, and totals are never "balanced" through rounding or epsilon
  comparisons.
* Blank or missing cells are never treated as zero. A sum relation that
  references such a cell is reported as indeterminate and left out of the
  search rather than silently evaluated.
* The original text and page coordinates (``bbox``) of every cell are
  preserved in the result.
"""

import enum
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from ._typing import T_bbox
from .table import Table

__all__ = [
    "NumberFormat",
    "Relation",
    "SumConstraint",
    "CellInput",
    "CellState",
    "CellResult",
    "ConstraintStatus",
    "ConstraintReport",
    "CleaningResult",
    "parse_candidates",
    "clean",
]

_Ref = Tuple[int, int]

_SIGN_CHARS = (("-", -1), ("\u2212", -1), ("+", 1))
_SPACE_CHARS = frozenset(
    [" ", "\u00a0", "\u2002", "\u2003", "\u2007", "\u2009", "\u202f"]
)
_DIAGNOSTIC_BUDGET = 100_000


class CellState(str, enum.Enum):
    """Classification of a table cell before constraint solving."""

    VALUE = "value"  # numeric text; parsed into >= 1 candidate
    BLANK = "blank"  # empty text (an existing but empty cell)
    MISSING = "missing"  # no cell at this grid position
    TEXT = "text"  # non-numeric text (no digits, e.g. a label)
    UNPARSEABLE = "unparseable"  # looks numeric but matches no declared format


class ConstraintStatus(str, enum.Enum):
    """Outcome of one sum constraint after cleaning."""

    ENFORCED = "enforced"  # active during solving; component is satisfiable
    INDETERMINATE = "indeterminate"  # references blank/missing cells; skipped
    INVALID = "invalid"  # references non-numeric cells; cannot hold
    CONFLICT = "conflict"  # jointly unsatisfiable with other constraints
    UNDECIDED = "undecided"  # search budget exhausted before a verdict


@dataclass(frozen=True)
class NumberFormat:
    """
    A declared regional number format.

    ``thousands`` is the thousands-group separator (``""`` if none),
    ``decimal`` the decimal separator, and ``symbols`` optional currency
    symbols or unit markers stripped from either end of the text before
    parsing (e.g. ``("$",)`` or ``("€",)``). Signs are always recognized:
    leading/trailing ``-``/``+`` (including U+2212) and parentheses for
    negatives, e.g. ``(1,234.56)``.

    Parsing is strict: thousands separators must appear in groups of three
    digits (``1,00`` is rejected), the decimal separator may appear at most
    once, and no other characters are allowed.
    """

    thousands: str = ","
    decimal: str = "."
    symbols: Tuple[str, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))
        if len(self.decimal) != 1:
            raise ValueError("decimal separator must be a single character")
        if len(self.thousands) > 1:
            raise ValueError("thousands separator must be empty or one character")
        if self.thousands and self.thousands == self.decimal:
            raise ValueError("thousands and decimal separators must differ")
        for sep in (self.thousands, self.decimal):
            if sep and (sep.isdecimal() or any(sep == c for c, _ in _SIGN_CHARS)):
                raise ValueError(f"invalid separator: {sep!r}")
        for sym in self.symbols:
            if not sym:
                raise ValueError("symbols must be non-empty strings")
            for ch in sym:
                if (
                    ch.isdecimal()
                    or any(ch == c for c, _ in _SIGN_CHARS)
                    or ch in (self.thousands, self.decimal)
                ):
                    raise ValueError(f"invalid character {ch!r} in symbol {sym!r}")

    @classmethod
    def en(cls, **kwargs: Any) -> "NumberFormat":
        """``1,000,000.00`` (US/UK-style)."""
        return cls(thousands=",", decimal=".", name="en", **kwargs)

    @classmethod
    def de(cls, **kwargs: Any) -> "NumberFormat":
        """``1.000.000,00`` (German-style)."""
        return cls(thousands=".", decimal=",", name="de", **kwargs)

    @classmethod
    def fr(cls, **kwargs: Any) -> "NumberFormat":
        """``1 000 000,00`` (French-style, space-grouped)."""
        return cls(thousands=" ", decimal=",", name="fr", **kwargs)

    @classmethod
    def ch(cls, **kwargs: Any) -> "NumberFormat":
        """``1'000'000.00`` (Swiss-style)."""
        return cls(thousands="'", decimal=".", name="ch", **kwargs)

    @classmethod
    def plain(cls, **kwargs: Any) -> "NumberFormat":
        """``1000000.00`` (no thousands separator)."""
        return cls(thousands="", decimal=".", name="plain", **kwargs)

    def parse(self, text: str) -> Optional[Decimal]:
        """
        Parse ``text`` under this format, returning an exact
        :class:`~decimal.Decimal`, or ``None`` if the text does not conform.
        """
        s = text.strip()
        if not s:
            return None
        negative = False
        if len(s) >= 2 and s.startswith("(") and s.endswith(")"):
            negative = True
            s = s[1:-1].strip()
        # Strip declared symbols and sign characters from either end,
        # in any order, e.g. "-$ 1,000" or "1.000,00 €".
        changed = True
        while changed:
            changed = False
            s = s.strip()
            for sym in self.symbols:
                if s.startswith(sym):
                    s = s[len(sym) :]
                    changed = True
                if s.endswith(sym):
                    s = s[: -len(sym)]
                    changed = True
            for char, sign in _SIGN_CHARS:
                if s.startswith(char):
                    negative = negative or sign < 0
                    s = s[1:]
                    changed = True
                if s.endswith(char):
                    negative = negative or sign < 0
                    s = s[:-1]
                    changed = True
        if not s:
            return None
        if self.thousands and self.thousands in _SPACE_CHARS:
            s = "".join(self.thousands if ch in _SPACE_CHARS else ch for ch in s)
        elif any(ch in _SPACE_CHARS for ch in s):
            return None
        int_part, sep, frac = s.partition(self.decimal)
        if sep and (not frac or not frac.isdecimal() or self.decimal in frac):
            return None
        if self.thousands and self.thousands in int_part:
            groups = int_part.split(self.thousands)
            if not groups[0] or len(groups[0]) > 3 or not groups[0].isdecimal():
                return None
            if any(len(g) != 3 or not g.isdecimal() for g in groups[1:]):
                return None
            int_digits = "".join(groups)
        elif int_part.isdecimal():
            int_digits = int_part
        else:
            return None
        try:
            value = Decimal(int_digits + ("." + frac if sep else ""))
        except InvalidOperation:  # pragma: nocover (defensive)
            return None
        return value.copy_negate() if negative else value


def parse_candidates(text: str, formats: Sequence[NumberFormat]) -> Tuple[Decimal, ...]:
    """
    All distinct exact values ``text`` can take under the declared formats,
    sorted ascending. ``"1,000"`` parsed under ``en`` and ``de`` yields
    ``(Decimal("1.000"), Decimal("1000"))``.
    """
    found: List[Decimal] = []
    for fmt in formats:
        value = fmt.parse(text)
        if value is not None and value not in found:
            found.append(value)
    return tuple(sorted(found))


@dataclass(frozen=True)
class CellInput:
    """One raw table cell: its extracted ``text`` and page ``bbox``."""

    text: Optional[str]
    bbox: Optional[T_bbox] = None


@dataclass(frozen=True)
class Relation:
    """
    A declared subtotal relation along one axis of the table.

    ``axis="row"``: ``total`` and ``addends`` are row indices, and the
    relation holds within each column — ``table[total][c]`` equals the sum
    of ``table[a][c]`` for ``a`` in ``addends``. ``axis="column"`` is the
    mirror image, holding within each row. ``within`` restricts the columns
    (``axis="row"``) or rows (``axis="column"``) the relation applies to;
    the default is all of them.
    """

    axis: str
    total: int
    addends: Tuple[int, ...]
    within: Optional[Tuple[int, ...]] = None
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "addends", tuple(self.addends))
        if self.within is not None:
            object.__setattr__(self, "within", tuple(self.within))
        if self.axis not in ("row", "column"):
            raise ValueError("axis must be 'row' or 'column'")
        if not self.addends:
            raise ValueError("a relation needs at least one addend")
        if len(set(self.addends)) != len(self.addends):
            raise ValueError("duplicate addend indices")
        if self.total in self.addends:
            raise ValueError("total index must not appear among addends")
        indices = (self.total,) + self.addends + (self.within or ())
        if any(i < 0 for i in indices):
            raise ValueError("indices must be non-negative")


@dataclass(frozen=True)
class SumConstraint:
    """
    A declared sum over explicit cells: the cell at ``total`` (a
    ``(row, column)`` pair) equals the sum of the cells at ``addends``.
    """

    total: _Ref
    addends: Tuple[_Ref, ...]
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "total", tuple(self.total))
        object.__setattr__(self, "addends", tuple(tuple(a) for a in self.addends))
        refs = (self.total,) + self.addends
        if any(len(r) != 2 for r in refs):
            raise ValueError("cell references must be (row, column) pairs")
        if not self.addends:
            raise ValueError("a constraint needs at least one addend")
        if len(set(self.addends)) != len(self.addends):
            raise ValueError("duplicate addend cells")
        if self.total in self.addends:
            raise ValueError("total cell must not appear among addends")
        if any(i < 0 for r in refs for i in r):
            raise ValueError("indices must be non-negative")


@dataclass(frozen=True)
class CellResult:
    """
    The cleaning outcome for one cell.

    ``text`` and ``bbox`` are the original extracted values, preserved
    verbatim. ``candidates`` are the parses under the declared formats;
    ``possible`` are the values still feasible after applying every sum
    relation. ``value`` is set only when exactly one value remains — i.e.
    the cell takes the same value in all feasible interpretations.
    """

    row: int
    col: int
    text: Optional[str]
    bbox: Optional[T_bbox]
    state: CellState
    candidates: Tuple[Decimal, ...]
    possible: Tuple[Decimal, ...]
    value: Optional[Decimal]
    constrained: bool

    @property
    def resolved(self) -> bool:
        """True when the cell has exactly one feasible value."""
        return self.value is not None

    @property
    def ambiguous(self) -> bool:
        """True when more than one value remains feasible."""
        return len(self.possible) > 1

    @property
    def conflicted(self) -> bool:
        """True when conflicting totals leave no feasible value."""
        return self.state is CellState.VALUE and self.constrained and not self.possible


@dataclass(frozen=True)
class ConstraintReport:
    """The fate of one declared sum relation."""

    status: ConstraintStatus
    total: _Ref
    addends: Tuple[_Ref, ...]
    label: str
    detail: str = ""
    satisfiable_alone: Optional[bool] = None


@dataclass(frozen=True, repr=False)
class CleaningResult:
    """
    The outcome of :func:`clean`.

    ``cells`` mirrors the table grid. ``reports`` covers every declared
    relation. ``complete`` is False when a search budget was exhausted; in
    that case ambiguous cells may list more ``possible`` values than truly
    feasible, but resolved ``value``s are still guaranteed unique across
    all feasible interpretations.
    """

    cells: Tuple[Tuple[CellResult, ...], ...]
    formats: Tuple[NumberFormat, ...]
    reports: Tuple[ConstraintReport, ...]
    complete: bool

    def __repr__(self) -> str:
        n_cells = sum(len(row) for row in self.cells)
        n_resolved = sum(1 for row in self.cells for c in row if c.resolved)
        return (
            f"CleaningResult(cells={n_cells}, resolved={n_resolved}, "
            f"conflicts={len(self.conflicts)}, complete={self.complete})"
        )

    @property
    def conflicts(self) -> Tuple[ConstraintReport, ...]:
        """Reports of totals that cannot be reconciled."""
        return tuple(
            r
            for r in self.reports
            if r.status in (ConstraintStatus.CONFLICT, ConstraintStatus.INVALID)
        )

    @property
    def indeterminate(self) -> Tuple[ConstraintReport, ...]:
        """Reports of relations skipped over blank/missing cells."""
        return tuple(
            r for r in self.reports if r.status is ConstraintStatus.INDETERMINATE
        )

    @property
    def resolved_cells(self) -> Tuple[CellResult, ...]:
        return tuple(c for row in self.cells for c in row if c.resolved)

    @property
    def ambiguous_cells(self) -> Tuple[CellResult, ...]:
        return tuple(c for row in self.cells for c in row if c.ambiguous)

    def cell(self, row: int, col: int) -> CellResult:
        return self.cells[row][col]

    def to_rows(self) -> List[List[Optional[Decimal]]]:
        """
        The persistable view: one grid of exact values, with ``None``
        wherever a cell is ambiguous, blank, missing, non-numeric, or
        caught in a conflict. Only values unique across all feasible
        interpretations are materialized.
        """
        return [[c.value for c in row] for row in self.cells]


# ---------------------------------------------------------------------
# Exact arithmetic helpers (no floats, no decimal-context rounding)
# ---------------------------------------------------------------------


def _to_int_pair(d: Decimal) -> Tuple[int, int]:
    """Return ``(mantissa, exponent)`` with ``d == mantissa * 10**exponent``."""
    t = d.as_tuple()
    mantissa = 0
    for digit in t.digits:
        mantissa = mantissa * 10 + digit
    if t.sign:
        mantissa = -mantissa
    return mantissa, cast(int, t.exponent)


def _from_int_pair(mantissa: int, exponent: int) -> Decimal:
    sign = 1 if mantissa < 0 else 0
    digits = tuple(int(c) for c in str(abs(mantissa)))
    return Decimal((sign, digits, exponent))


def _combine(terms: Iterable[Tuple[Decimal, int]]) -> Decimal:
    """Exact sum of ``sign * value`` terms, computed with integers."""
    pairs = []
    for value, sign in terms:
        mantissa, exponent = _to_int_pair(value)
        pairs.append((mantissa * sign, exponent))
    exponent = min(e for _, e in pairs)
    total = sum(m * 10 ** (e - exponent) for m, e in pairs)
    return _from_int_pair(total, exponent)


# ---------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------


@dataclass
class _NormConstraint:
    total: _Ref
    addends: Tuple[_Ref, ...]
    label: str
    origin: object

    @property
    def refs(self) -> Tuple[_Ref, ...]:
        return (self.total,) + self.addends


class _BudgetExceeded(Exception):
    pass


def _propagate(
    constraints: Sequence[_NormConstraint],
    domains: Dict[_Ref, Tuple[Decimal, ...]],
    assignment: Dict[_Ref, Decimal],
    record: Callable[[_Ref], None],
) -> bool:
    """
    Enforce constraints with at most one unassigned cell. Mutates
    ``assignment``, reporting each new assignment via ``record``.
    Returns False on contradiction.
    """
    changed = True
    while changed:
        changed = False
        for con in constraints:
            missing = [r for r in con.refs if r not in assignment]
            if not missing:
                summed = _combine((assignment[a], 1) for a in con.addends)
                if summed != assignment[con.total]:
                    return False
            elif len(missing) == 1:
                ref = missing[0]
                if ref == con.total:
                    required = _combine((assignment[a], 1) for a in con.addends)
                else:
                    terms = [(assignment[con.total], 1)]
                    terms += [(assignment[a], -1) for a in con.addends if a != ref]
                    required = _combine(terms)
                if required not in domains[ref]:
                    return False
                # assign the domain's own candidate, so solved cells keep
                # the exact parsed value (e.g. 1000, not 1000.0)
                assignment[ref] = domains[ref][domains[ref].index(required)]
                record(ref)
                changed = True
    return True


def _pick_unassigned(
    domains: Dict[_Ref, Tuple[Decimal, ...]], assignment: Dict[_Ref, Decimal]
) -> Optional[_Ref]:
    best: Optional[_Ref] = None
    for ref, domain in domains.items():
        if ref not in assignment and (best is None or len(domain) < len(domains[best])):
            best = ref
    return best


def _search(
    domains: Dict[_Ref, Tuple[Decimal, ...]],
    constraints: Sequence[_NormConstraint],
    node_budget: int,
    solution_limit: int,
    seed: Optional[Dict[_Ref, Decimal]] = None,
) -> Tuple[List[Dict[_Ref, Decimal]], bool]:
    """
    Depth-first search for assignments satisfying every constraint exactly.

    Returns ``(solutions, exhausted)``; ``exhausted`` is False when the
    search stopped at ``solution_limit``. Raises :class:`_BudgetExceeded`
    when ``node_budget`` assignment attempts are spent.
    """
    assignment: Dict[_Ref, Decimal] = dict(seed or {})
    root_trail: List[_Ref] = []
    if not _propagate(constraints, domains, assignment, root_trail.append):
        return [], True

    solutions: List[Dict[_Ref, Decimal]] = []
    frames: List[List[Any]] = []  # each: [ref, remaining values, trail]
    nodes = 0

    while True:
        ref = _pick_unassigned(domains, assignment)
        if ref is None:
            solutions.append(dict(assignment))
            if len(solutions) >= solution_limit:
                return solutions, False
        else:
            frames.append([ref, list(domains[ref]), []])

        advanced = False
        while frames:
            frame = frames[-1]
            for assigned_ref in frame[2]:
                assignment.pop(assigned_ref, None)
            frame[2] = []
            if not frame[1]:
                frames.pop()
                continue
            value = frame[1].pop()
            nodes += 1
            if nodes > node_budget:
                raise _BudgetExceeded()
            assignment[frame[0]] = value
            frame[2].append(frame[0])
            if _propagate(constraints, domains, assignment, frame[2].append):
                advanced = True
                break
        if not advanced:
            return solutions, True


@dataclass
class _ComponentOutcome:
    status: str  # "ok" | "unsat" | "undecided"
    complete: bool
    possible: Dict[_Ref, Set[Decimal]]


def _solve_component(
    refs: List[_Ref],
    domains: Dict[_Ref, Tuple[Decimal, ...]],
    constraints: List[_NormConstraint],
    node_budget: int,
    solution_limit: int,
) -> _ComponentOutcome:
    sub_domains = {r: domains[r] for r in refs}
    try:
        solutions, exhausted = _search(
            sub_domains, constraints, node_budget, solution_limit
        )
    except _BudgetExceeded:
        return _ComponentOutcome(
            "undecided", False, {r: set(sub_domains[r]) for r in refs}
        )
    possible: Dict[_Ref, Set[Decimal]] = {r: set() for r in refs}
    if not solutions:
        return _ComponentOutcome("unsat", True, possible)
    for solution in solutions:
        for ref, value in solution.items():
            possible[ref].add(value)
    if exhausted:
        return _ComponentOutcome("ok", True, possible)
    # The solution limit was hit: test each unseen candidate individually
    # so that ``possible`` stays exact where the budget allows.
    complete = True
    for ref in refs:
        for candidate in sub_domains[ref]:
            if candidate in possible[ref]:
                continue
            try:
                extra, _ = _search(
                    sub_domains, constraints, node_budget, 1, seed={ref: candidate}
                )
            except _BudgetExceeded:
                possible[ref].add(candidate)  # conservative: keep the value
                complete = False
                continue
            if extra:
                possible[ref].add(candidate)
    return _ComponentOutcome("ok", complete, possible)


def _satisfiable_alone(
    constraint: _NormConstraint, domains: Dict[_Ref, Tuple[Decimal, ...]]
) -> Optional[bool]:
    sub_domains = {r: domains[r] for r in constraint.refs}
    try:
        solutions, _ = _search(sub_domains, [constraint], _DIAGNOSTIC_BUDGET, 1)
    except _BudgetExceeded:
        return None
    return bool(solutions)


def _components(
    active: List[_NormConstraint],
) -> List[Tuple[List[_Ref], List[_NormConstraint]]]:
    """Group refs and constraints into connected components (union-find)."""
    parent: Dict[_Ref, _Ref] = {}

    def find(x: _Ref) -> _Ref:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for con in active:
        first = con.refs[0]
        for ref in con.refs[1:]:
            ra, rb = find(first), find(ref)
            if ra != rb:
                parent[rb] = ra

    all_refs: Set[_Ref] = set()
    for con in active:
        all_refs.update(con.refs)
    refs_by_root: Dict[_Ref, List[_Ref]] = {}
    for ref in sorted(all_refs):
        refs_by_root.setdefault(find(ref), []).append(ref)
    cons_by_root: Dict[_Ref, List[_NormConstraint]] = {}
    for con in active:
        cons_by_root.setdefault(find(con.refs[0]), []).append(con)
    return [(refs_by_root[root], cons_by_root[root]) for root in refs_by_root]


# ---------------------------------------------------------------------
# Grid normalization
# ---------------------------------------------------------------------

_CellLike = Union[None, str, CellInput, Tuple[Optional[str], Optional[T_bbox]]]


def _coerce_cell(item: _CellLike) -> CellInput:
    if item is None:
        return CellInput(text=None, bbox=None)
    if isinstance(item, CellInput):
        return item
    if isinstance(item, str):
        return CellInput(text=item, bbox=None)
    if isinstance(item, (tuple, list)) and len(item) == 2:
        text, bbox = item
        if text is not None and not isinstance(text, str):
            raise TypeError(f"cell text must be str or None, got {text!r}")
        if bbox is not None and len(bbox) != 4:
            raise ValueError(f"cell bbox must have 4 numbers, got {bbox!r}")
        return CellInput(text=text, bbox=bbox)
    raise TypeError(f"cannot interpret table cell: {item!r}")


def _grid_from_table(
    table: Table, text_settings: Optional[Dict[str, Any]]
) -> List[List[CellInput]]:
    texts = table.extract(**(text_settings or {}))
    grid = []
    for r, row in enumerate(table.rows):
        text_row = texts[r] if r < len(texts) else []
        line = []
        for c, bbox in enumerate(row.cells):
            text = text_row[c] if c < len(text_row) else None
            line.append(CellInput(text=text, bbox=bbox))
        grid.append(line)
    return grid


def _classify(
    text: Optional[str], formats: Tuple[NumberFormat, ...]
) -> Tuple[CellState, Tuple[Decimal, ...]]:
    if text is None:
        return CellState.MISSING, ()
    stripped = text.strip()
    if not stripped:
        return CellState.BLANK, ()
    candidates = parse_candidates(stripped, formats)
    if candidates:
        return CellState.VALUE, candidates
    if any(ch.isdecimal() for ch in stripped):
        return CellState.UNPARSEABLE, ()
    return CellState.TEXT, ()


def _check_bounds(indices: Iterable[int], dimension: int, origin: object) -> None:
    for index in indices:
        if index >= dimension:
            raise ValueError(
                f"index {index} out of range for grid dimension "
                f"{dimension} in {origin!r}"
            )


def _expand_relation(
    relation: Relation, n_rows: int, n_cols: int
) -> List[_NormConstraint]:
    out = []
    if relation.axis == "row":
        _check_bounds((relation.total,) + relation.addends, n_rows, relation)
        span = relation.within if relation.within is not None else tuple(range(n_cols))
        _check_bounds(span, n_cols, relation)
        for col in span:
            label = relation.label or (
                f"row {relation.total} = sum of rows "
                f"{list(relation.addends)} in column {col}"
            )
            if relation.label:
                label = f"{relation.label} (column {col})"
            out.append(
                _NormConstraint(
                    total=(relation.total, col),
                    addends=tuple((a, col) for a in relation.addends),
                    label=label,
                    origin=relation,
                )
            )
    else:
        _check_bounds((relation.total,) + relation.addends, n_cols, relation)
        span = relation.within if relation.within is not None else tuple(range(n_rows))
        _check_bounds(span, n_rows, relation)
        for row in span:
            label = relation.label or (
                f"column {relation.total} = sum of columns "
                f"{list(relation.addends)} in row {row}"
            )
            if relation.label:
                label = f"{relation.label} (row {row})"
            out.append(
                _NormConstraint(
                    total=(row, relation.total),
                    addends=tuple((row, a) for a in relation.addends),
                    label=label,
                    origin=relation,
                )
            )
    return out


def clean(
    source: Union[Table, Sequence[Sequence[_CellLike]]],
    formats: Sequence[NumberFormat],
    relations: Sequence[Union[Relation, SumConstraint]] = (),
    *,
    text_settings: Optional[Dict[str, Any]] = None,
    solution_limit: int = 4096,
    node_budget: int = 200_000,
) -> CleaningResult:
    """
    Clean locale-ambiguous amounts in an extracted table.

    :param source: a :class:`pdfplumber.table.Table` (its native cell text
        and bboxes are used), or a grid of raw cells. Each raw cell may be
        ``None`` (missing), a ``str``, a ``(text, bbox)`` pair, or a
        :class:`CellInput`. Ragged rows are padded with missing cells.
    :param formats: the declared :class:`NumberFormat`\\ s each cell's text
        is parsed under. Every format contributes at most one candidate per
        cell.
    :param relations: known subtotal structure, as :class:`Relation`
        (row/column-wise) and/or :class:`SumConstraint` (explicit cells)
        objects.
    :param text_settings: forwarded to ``Table.extract`` when ``source``
        is a table.
    :param solution_limit: maximum interpretations enumerated per
        connected component before falling back to per-candidate checks.
    :param node_budget: maximum solver search steps per component.

    The result keeps each cell's original text and page ``bbox``. Values
    are materialized (``CellResult.value`` / :meth:`CleaningResult.to_rows`)
    only when unique across all feasible interpretations; ambiguous cells
    carry ``possible`` values; unreconcilable totals appear in
    ``result.conflicts``. Blank and missing cells are never treated as
    zero, and no floating-point approximation is used anywhere.
    """
    fmts = tuple(dict.fromkeys(formats))
    if not fmts:
        raise ValueError("declare at least one NumberFormat")
    if solution_limit < 1:
        raise ValueError("solution_limit must be >= 1")
    if node_budget < 1:
        raise ValueError("node_budget must be >= 1")

    if isinstance(source, Table):
        raw = _grid_from_table(source, text_settings)
    else:
        raw = [[_coerce_cell(item) for item in row] for row in source]
    n_cols = max((len(row) for row in raw), default=0)
    grid = [row + [CellInput(None, None)] * (n_cols - len(row)) for row in raw]
    n_rows = len(grid)

    states: Dict[_Ref, CellState] = {}
    domains: Dict[_Ref, Tuple[Decimal, ...]] = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            state, candidates = _classify(cell.text, fmts)
            states[(r, c)] = state
            domains[(r, c)] = candidates

    norm: List[_NormConstraint] = []
    for relation in relations:
        if isinstance(relation, Relation):
            norm.extend(_expand_relation(relation, n_rows, n_cols))
        elif isinstance(relation, SumConstraint):
            refs = (relation.total,) + relation.addends
            for ref in refs:
                if ref[0] >= n_rows or ref[1] >= n_cols:
                    raise ValueError(
                        f"cell {ref} out of range for grid "
                        f"{n_rows}x{n_cols} in {relation!r}"
                    )
            label = relation.label or (
                f"cell {relation.total} = sum of {list(relation.addends)}"
            )
            norm.append(
                _NormConstraint(
                    total=relation.total,
                    addends=relation.addends,
                    label=label,
                    origin=relation,
                )
            )
        else:
            raise TypeError(
                "relations must be Relation or SumConstraint, "
                f"got {type(relation).__name__}"
            )

    deduped: List[_NormConstraint] = []
    seen_keys: Set[Tuple[_Ref, FrozenSet[_Ref]]] = set()
    for con in norm:
        key = (con.total, frozenset(con.addends))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(con)

    reports: List[ConstraintReport] = []
    active: List[_NormConstraint] = []
    for con in deduped:
        blank_refs = tuple(
            r for r in con.refs if states[r] in (CellState.BLANK, CellState.MISSING)
        )
        nonnumeric_refs = tuple(
            r for r in con.refs if states[r] in (CellState.TEXT, CellState.UNPARSEABLE)
        )
        if blank_refs:
            reports.append(
                ConstraintReport(
                    ConstraintStatus.INDETERMINATE,
                    con.total,
                    con.addends,
                    con.label,
                    detail=(
                        "references blank/missing cells, which never count "
                        f"as zero: {list(blank_refs)}"
                    ),
                )
            )
        elif nonnumeric_refs:
            reports.append(
                ConstraintReport(
                    ConstraintStatus.INVALID,
                    con.total,
                    con.addends,
                    con.label,
                    detail=f"references non-numeric cells: {list(nonnumeric_refs)}",
                )
            )
        else:
            active.append(con)

    complete = True
    ref_outcome: Dict[_Ref, _ComponentOutcome] = {}
    for comp_refs, comp_constraints in _components(active):
        outcome = _solve_component(
            comp_refs, domains, comp_constraints, node_budget, solution_limit
        )
        for ref in comp_refs:
            ref_outcome[ref] = outcome
        if outcome.status == "unsat":
            for con in comp_constraints:
                reports.append(
                    ConstraintReport(
                        ConstraintStatus.CONFLICT,
                        con.total,
                        con.addends,
                        con.label,
                        detail=(
                            "no assignment of the declared format "
                            "interpretations satisfies all totals "
                            "simultaneously"
                        ),
                        satisfiable_alone=_satisfiable_alone(con, domains),
                    )
                )
        else:
            if not outcome.complete:
                complete = False
            status = (
                ConstraintStatus.ENFORCED
                if outcome.status == "ok"
                else ConstraintStatus.UNDECIDED
            )
            detail = (
                ""
                if outcome.complete
                else "search budget exhausted; cell values are conservative"
            )
            for con in comp_constraints:
                reports.append(
                    ConstraintReport(
                        status, con.total, con.addends, con.label, detail=detail
                    )
                )

    result_rows: List[Tuple[CellResult, ...]] = []
    for r in range(n_rows):
        line: List[CellResult] = []
        for c in range(n_cols):
            ref = (r, c)
            state = states[ref]
            candidates = domains[ref]
            cell_outcome = ref_outcome.get(ref)
            if state is CellState.VALUE:
                if cell_outcome is None:
                    possible: Tuple[Decimal, ...] = candidates
                elif cell_outcome.status == "unsat":
                    possible = ()
                else:
                    possible = tuple(sorted(cell_outcome.possible[ref]))
            else:
                possible = ()
            value = possible[0] if len(possible) == 1 else None
            line.append(
                CellResult(
                    row=r,
                    col=c,
                    text=grid[r][c].text,
                    bbox=grid[r][c].bbox,
                    state=state,
                    candidates=candidates,
                    possible=possible,
                    value=value,
                    constrained=cell_outcome is not None,
                )
            )
        result_rows.append(tuple(line))

    return CleaningResult(
        cells=tuple(result_rows),
        formats=fmts,
        reports=tuple(reports),
        complete=complete,
    )
