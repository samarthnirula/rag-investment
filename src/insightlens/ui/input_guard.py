"""Input validation and prompt-injection defence.

Validates user query strings before they reach the LLM or retrieval pipeline.
Raises InputGuardError on any violation so the UI can show a clean message.
"""
from __future__ import annotations

import re
import unicodedata

_MAX_QUERY_LENGTH = 2000
_MIN_QUERY_LENGTH = 2

# Prompt-injection attempts
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+\w+", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you|above)", re.IGNORECASE),
    re.compile(r"(reveal|print|output|repeat|show)\s+(your\s+)?(system\s+prompt|instructions?|api\s+key|password|credentials?)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a|an)\s+(?:different|new|another|unrestricted)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"new\s+conversation\s+start", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<\|system\|>", re.IGNORECASE),
    re.compile(r"###\s*(system|instruction|prompt)", re.IGNORECASE),
    re.compile(r"(?:pretend|roleplay|simulate)\s+(?:you\s+are|being|as\s+if)", re.IGNORECASE),
    re.compile(r"override\s+(?:your\s+)?(?:safety|guardrail|filter|restriction)", re.IGNORECASE),
]

# Strip control characters and null bytes
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Repeated character spam (e.g. "aaaaaaa..." or "...........")
_SPAM_RE = re.compile(r"(.)\1{60,}")

# HTML/script injection patterns (for display safety in discussion posts).
# Blocks: <script, <img, <iframe, <object, <embed, <svg, <link, <meta,
#         javascript:, data: URIs, event handlers (on*=), CSS expression().
_HTML_SCRIPT_RE = re.compile(
    r"<\s*(?:script|img|iframe|object|embed|svg|link|meta|form|input|button)"
    r"|javascript\s*:"
    r"|data\s*:\s*(?:text|application|image)"
    r"|on\w+\s*="
    r"|expression\s*\(",
    re.IGNORECASE,
)


class InputGuardError(ValueError):
    """Raised when a query fails validation — message is safe to show the user."""


def _normalize(text: str) -> str:
    """Normalize unicode to NFKC and strip control characters."""
    text = unicodedata.normalize("NFKC", text)
    return _CONTROL_RE.sub("", text).strip()


def validate_query(query: str) -> str:
    """Validate and sanitize a user query for the LLM/retrieval pipeline.

    Returns the cleaned query. Raises InputGuardError on any violation.
    """
    if not query or not query.strip():
        raise InputGuardError("Query cannot be empty.")

    query = _normalize(query)

    if len(query) < _MIN_QUERY_LENGTH:
        raise InputGuardError("Query is too short. Please ask a complete question.")

    if len(query) > _MAX_QUERY_LENGTH:
        raise InputGuardError(
            f"Query is too long ({len(query):,} characters). "
            f"Please keep questions under {_MAX_QUERY_LENGTH:,} characters."
        )

    if _SPAM_RE.search(query):
        raise InputGuardError("Query contains repeated characters. Please ask a real question.")

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            raise InputGuardError(
                "Your message was flagged as a potential system manipulation attempt. "
                "Please rephrase your legal research question."
            )

    return query


def validate_text_input(text: str, field: str = "Input", max_length: int = 500) -> str:
    """Lighter validation for free-text fields (names, post content).

    Strips control chars, checks length, blocks script injection.
    Does NOT apply prompt-injection rules (those are LLM-specific).
    """
    if not text or not text.strip():
        raise InputGuardError(f"{field} cannot be empty.")

    text = _normalize(text)

    if len(text) > max_length:
        raise InputGuardError(
            f"{field} is too long ({len(text):,} characters). "
            f"Maximum {max_length:,} characters."
        )

    if _HTML_SCRIPT_RE.search(text):
        raise InputGuardError(f"{field} contains disallowed content.")

    return text
