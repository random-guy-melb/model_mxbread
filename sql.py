
from typing import Iterable, Sequence
import re


def dequote_columns(where_clause: str, columns: Iterable[str]) -> str:
    """
    Remove leading/trailing `"` or `'` around the *given* column names
    wherever they occur as standalone identifiers (optionally qualified
    with dots), without touching quotes that belong to string literals.

    Parameters
    ----------
    where_clause : str
        Raw WHERE-clause string from the LLM.
    columns : Iterable[str]
        Exact column names to unquote (case-sensitive).

    Returns
    -------
    str
        The cleaned WHERE clause.
    """
    # Compile one regex that can match *all* supplied identifiers.
    pattern = _build_pattern(columns)
    return pattern.sub(_make_replacer(), where_clause)

def _build_pattern(columns: Sequence[str]) -> re.Pattern[str]:
    """
    Build a single compiled regex that matches any of the supplied columns
    wrapped in optional quotes, preceded by a word-separating char
    (^, space, (, comma, dot) and *followed* by either an operator or a ')'.

    Supporting a right-paren lets us handle function calls such as::

        DATE_TRUNC('day', "timestamp")
    """
    if not columns:
        raise ValueError("columns must contain at least one identifier")

    # Pre-escape the identifiers and OR-join them
    identifier_alts = "|".join(map(re.escape, columns))

    # Common comparison keywords/operators you expect right after an identifier
    operators = (
        r"=|!=|<>|>=|<=|>|<|LIKE|ILIKE|IN|IS|NOT|BETWEEN|AND|OR"
    )

    # Regex, verbose mode for clarity
    pattern = rf"""
        (?P<prefix>^|[(\s,\.])          # ① left boundary (start / ( / space / , / .)
        (?P<openquote>["']?)            # ② optional opening quote
        (?P<identifier>{identifier_alts})  # ③ one of our columns
        (?P=openquote)?                # ④ matching closing quote (if any)
        (?P<suffix>                    # ⑤ we keep what follows intact
            \s*(?:                     #    either…
                (?:{operators})\b      #    …a comparison operator/keyword
              | \)                     #    …or a right-paren (func arg)
            )
        )
    """

    return re.compile(pattern, flags=re.IGNORECASE | re.VERBOSE)


def _make_replacer():
    """
    Return a tiny closure so `re.sub` can efficiently drop the quotes
    while preserving the prefix/suffix groups.
    """

    def _replacer(match: re.Match[str]) -> str:  # noqa: WPS430
        return f"{match.group('prefix')}{match.group('identifier')}{match.group('suffix')}"

    return _replacer


if __name__ == "__main__":
    example = (
        '"group" ILIKE \'%network%\' AND "cause" = \'Software\' '
        'AND "priority" = \'Very High\' AND "timestamp" >= 12121212 '
        'AND evt."timestamp" < 31313214 '
        'AND DATE_TRUNC(\'day\', "timestamp") >= DATE_TRUNC(\'day\', NOW())'
    )

    cols = ["group", "cause", "priority", "timestamp"]
    cleaned = dequote_columns(example, cols)

    print("RAW CLAUSE:\n-----------")
    print(example)
    print("\nCLEANED:\n--------")
    print(cleaned)
