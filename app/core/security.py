"""OAuth2/JWT primitives and password hashing.

JWT bearer tokens carry ``sub`` (user id), ``tenant_id``, and ``role`` so
every request can be tenant-scoped without extra lookups beyond the user
check.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from pydantic import BaseModel, ConfigDict

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# scrypt parameters (OWASP minimums: N=2^14, r=8, p=1).
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_MIN_PASSWORD_LENGTH = 8


class AuthTokenError(Exception):
    """Raised for missing/invalid/expired tokens."""


class TokenPayload(BaseModel):
    """Validated JWT claims."""

    model_config = ConfigDict(extra="ignore")

    sub: str
    tenant_id: str
    role: str = "member"
    exp: datetime | None = None
    iat: datetime | None = None


# ── Passwords (stdlib scrypt — memory-hard, no legacy deps) ────────────
def hash_password(password: str) -> str:
    """Hash a plaintext password.

    Produces an self-describing PHC-style string:
    ``scrypt$ln=14,r=8,p=1$<salt_hex>$<hash_hex>`` so parameters can be
    upgraded later without breaking existing hashes.
    """
    if len(password) < _MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {_MIN_PASSWORD_LENGTH} characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    ln = _SCRYPT_N.bit_length() - 1
    return f"scrypt$ln={ln},r={_SCRYPT_R},p={_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(plain: str, hashed: str | None) -> bool:
    """Constant-time password verification against a stored hash."""
    if not hashed:
        return False
    try:
        algorithm, params, salt_hex, digest_hex = hashed.split("$")
        if algorithm != "scrypt":
            logger.warning("Unsupported password hash algorithm: %s", algorithm)
            return False
        kv = dict(pair.split("=", 1) for pair in params.split(","))
        computed = hashlib.scrypt(
            plain.encode(),
            salt=bytes.fromhex(salt_hex),
            n=2 ** int(kv["ln"]),
            r=int(kv["r"]),
            p=int(kv["p"]),
            dklen=len(bytes.fromhex(digest_hex)),
        )
        return hmac.compare_digest(computed.hex(), digest_hex)
    except (ValueError, KeyError):
        logger.warning("Malformed password hash encountered")
        return False


# ── Tokens ──────────────────────────────────────────────────────────────
def create_access_token(
    *,
    subject: str,
    tenant_id: str,
    role: str = "member",
    expires_minutes: int | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Issue a signed JWT bearer token."""
    settings = get_settings()
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=expires_minutes or settings.access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "tenant_id": tenant_id,
        "role": role,
        "iat": now,
        "exp": expire,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> TokenPayload:
    """Decode + validate a JWT; raise :class:`AuthTokenError` on failure."""
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        return TokenPayload(**claims)
    except JWTError as exc:
        raise AuthTokenError(f"Invalid token: {exc}") from exc
