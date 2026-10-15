"""Unit tests for Fernet credential encryption primitives."""

from __future__ import annotations

import pytest

from app.core.crypto import (
    DecryptionError,
    TokenCipher,
    clear_cipher_cache,
    generate_fernet_key,
    get_token_cipher,
    mask_token,
)


def test_encrypt_decrypt_roundtrip():
    cipher = TokenCipher(generate_fernet_key())
    token = "ya29.a0AfH6SMB-secret-refresh-token"
    ciphertext = cipher.encrypt(token)

    assert ciphertext != token
    assert cipher.decrypt(ciphertext) == token


def test_encryption_is_non_deterministic():
    """Fernet salts each encryption — same plaintext, different ciphertext."""
    cipher = TokenCipher(generate_fernet_key())
    assert cipher.encrypt("token") != cipher.encrypt("token")


def test_encrypt_empty_rejected():
    cipher = TokenCipher(generate_fernet_key())
    with pytest.raises(ValueError, match="empty"):
        cipher.encrypt("")


def test_decrypt_empty_rejected():
    cipher = TokenCipher(generate_fernet_key())
    with pytest.raises(DecryptionError):
        cipher.decrypt("")


def test_decrypt_with_wrong_key_raises():
    good = TokenCipher(generate_fernet_key())
    other = TokenCipher(generate_fernet_key())
    ciphertext = good.encrypt("secret")

    with pytest.raises(DecryptionError, match="decryption failed"):
        other.decrypt(ciphertext)


def test_decrypt_garbage_raises():
    cipher = TokenCipher(generate_fernet_key())
    with pytest.raises(DecryptionError):
        cipher.decrypt("not-a-fernet-token!!")


def test_invalid_key_rejected():
    with pytest.raises(ValueError, match="Invalid Fernet key"):
        TokenCipher("too-short")


def test_from_settings_uses_configured_key(settings):
    cipher = TokenCipher.from_settings()
    assert cipher.decrypt(cipher.encrypt("x")) == "x"
    assert settings.fernet_key.get_secret_value()


def test_token_cipher_is_cached_and_clearable():
    first = get_token_cipher()
    assert get_token_cipher() is first
    clear_cipher_cache()
    assert get_token_cipher() is not first


def test_mask_token():
    assert mask_token("") == ""
    assert mask_token("abcd") == "****"  # too short to reveal anything
    masked = mask_token("secret-token-value-1234")
    assert masked.startswith("secr") and masked.endswith("1234")
    assert "token" not in masked
