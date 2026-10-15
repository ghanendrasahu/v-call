"""Symmetric encryption primitives for credential storage.

Integration tokens (Zoho, Freshdesk, Jira, Notion, Google…) are encrypted at
rest with Fernet (AES-128-CBC + HMAC-SHA256). Raw plaintext never touches the
database or the logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings


class DecryptionError(Exception):
    """Raised when a stored ciphertext cannot be decrypted."""


def generate_fernet_key() -> str:
    """Generate a new url-safe base64 Fernet key."""
    return Fernet.generate_key().decode()


class TokenCipher:
    """Thin, side-effect-free wrapper around :class:`cryptography.fernet.Fernet`."""

    __slots__ = ("_fernet",)

    def __init__(self, key: str) -> None:
        try:
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except (ValueError, TypeError) as exc:  # malformed key
            raise ValueError("Invalid Fernet key: must be 32 url-safe base64 bytes") from exc

    @classmethod
    def from_settings(cls) -> TokenCipher:
        return cls(get_settings().fernet_key.get_secret_value())

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a UTF-8 string, returning url-safe base64 ciphertext."""
        if not plaintext:
            raise ValueError("Cannot encrypt an empty token")
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext, raising :class:`DecryptionError` on failure."""
        if not ciphertext:
            raise DecryptionError("Empty ciphertext")
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise DecryptionError(
                "Token decryption failed (key rotated or data corrupted)"
            ) from exc


@lru_cache
def get_token_cipher() -> TokenCipher:
    """Process-wide cipher bound to the configured Fernet key."""
    return TokenCipher.from_settings()


def clear_cipher_cache() -> None:
    """Drop the cached cipher (intended for tests)."""
    get_token_cipher.cache_clear()


def mask_token(token: str, visible: int = 4) -> str:
    """Mask a secret for safe display/logging: ``zoho_abc…wxyz``."""
    if not token:
        return ""
    if len(token) <= visible * 2:
        return "*" * len(token)
    return f"{token[:visible]}…{token[-visible:]}"


@dataclass(frozen=True, slots=True)
class DecryptedCredential:
    """A fully decrypted integration credential (never log this object)."""

    provider: str
    access_token: str
    refresh_token: str | None
