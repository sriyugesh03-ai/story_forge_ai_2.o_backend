import hashlib
import secrets


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 (kept for legacy reference)."""
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return f"{salt.hex()}:{key.hex()}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its PBKDF2 hash."""
    try:
        salt_hex, key_hex = hashed_password.split(':')
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)

        test_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return secrets.compare_digest(test_key, expected_key)
    except Exception:
        return False