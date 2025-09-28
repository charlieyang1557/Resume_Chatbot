"""Shared text processing helpers for the resume chatbot."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

# Alias groups for the candidate's different names. All tokens are stored in
# lowercase to simplify matching logic across the application.
_CANDIDATE_ALIAS_GROUPS: Sequence[frozenset[str]] = (
    frozenset({"charlie", "yutian", "yang"}),
)

# Build a lookup so we can quickly expand any token into its alias group.
_ALIAS_LOOKUP: dict[str, tuple[str, ...]] = {
    alias: tuple(sorted(group))
    for group in _CANDIDATE_ALIAS_GROUPS
    for alias in group
}


def expand_aliases(tokens: Iterable[str]) -> List[str]:
    """Expand tokens with any known aliases and deduplicate the result.

    Args:
        tokens: Iterable of raw tokens (already normalised to lowercase).

    Returns:
        List of tokens including any alias equivalents, preserving order.
    """

    expanded: List[str] = []
    for token in tokens:
        lookup_key = token.lower()
        aliases = _ALIAS_LOOKUP.get(lookup_key)
        if aliases:
            expanded.extend(aliases)
        else:
            expanded.append(lookup_key)

    seen: set[str] = set()
    deduplicated: List[str] = []
    for token in expanded:
        if token not in seen:
            seen.add(token)
            deduplicated.append(token)
    return deduplicated


def tokenize_with_aliases(
    text: str,
    *,
    stopwords: set[str] | frozenset[str] | None = None,
    pattern: re.Pattern[str] | None = None,
) -> List[str]:
    """Tokenise ``text`` and expand candidate aliases.

    Args:
        text: Input string to tokenize.
        stopwords: Optional collection of stopwords to filter out.
        pattern: Optional precompiled regex pattern used for tokenisation.

    Returns:
        List of lowercase tokens with aliases expanded.
    """

    regex = pattern or re.compile(r"\b\w+\b")
    raw_tokens = [token.lower() for token in regex.findall(text)]
    if stopwords:
        raw_tokens = [token for token in raw_tokens if token not in stopwords]
    return expand_aliases(raw_tokens)


__all__ = [
    "expand_aliases",
    "tokenize_with_aliases",
]
