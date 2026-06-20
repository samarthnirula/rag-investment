"""Application-layer encryption for secrets stored at rest (OAuth tokens, etc.).

Uses Fernet (AES-128-CBC + HMAC, from the `cryptography` package, which is
already a transitive dependency via pyjwt[crypto]). The key is read from the
TOKEN_ENCRYPTION_KEY environment variable -- a urlsafe-base64, 32-byte key as
produced by `Fernet.generate_key()`.

This closes the gap flagged in migrations/003_cloud_credentials.sql: that
table's COMMENT previously claimed tokens were "encrypted" when the column
was plain TEXT. Ciphertext produced here is what actually gets written to
refresh_token / access_token now.
"""
from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken

_ENV_VAR = "TOKEN_ENCRYPTION_KEY"


class TokenCryptoError(Exception):
    """Raised when encryption/decryption cannot proceed."""


def _get_fernet() -> Fernet:
    key = os.getenv(_ENV_VAR, "").strip()
    if not key:
        raise TokenCryptoError(
            f"{_ENV_VAR} is not set. Generate one with "
            f"`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            f"and add it to .env. Cloud storage OAuth tokens cannot be stored or read without it."
        )
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as exc:
        raise TokenCryptoError(
            f"{_ENV_VAR} is not a valid Fernet key (must be 32 url-safe base64 bytes)."
        ) from exc


def encrypt_token(plaintext: str | None) -> str | None:
    """Encrypt a token for storage. None passes through unchanged."""
    if plaintext is None:
        return None
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str | None) -> str | None:
    """Decrypt a token read from storage. None passes through unchanged."""
    if ciphertext is None:
        return None
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise TokenCryptoError(
            "Stored token could not be decrypted -- wrong/rotated "
            "TOKEN_ENCRYPTION_KEY, or the value predates encryption being enabled."
        ) from exc
