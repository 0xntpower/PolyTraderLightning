"""Cross-platform secure credential storage for PolySignalLab.

Lookup order for all secrets:
  1. Windows Credential Manager (if on Windows and a credential exists).
  2. Age-encrypted secrets file (if ``age`` binary and SSH host key exist).
  3. Environment variable (deprecated legacy fallback).

On Windows, credentials are encrypted at rest via DPAPI (tied to the
user account).  On Linux, secrets are encrypted with ``age`` using the
machine's SSH host key — no password prompt required on startup.
Environment variables / ``.env`` files remain as a deprecated fallback
for environments where neither backend is available.

All functions in this module are safe to call from any platform.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_CRED_PREFIX = "PolySignalLab/"
_HMAC_CRED_TARGET = "PolySignalLab/HMAC_KEY"
_HMAC_ENV_KEY = "PLSLAB_HMAC_KEY"
_MIN_HMAC_KEY_BYTES = 32  # 256-bit minimum

_AGE_SUBPROCESS_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Age backend error
# ---------------------------------------------------------------------------


class AgeBackendError(RuntimeError):
    """Raised when the age encryption backend encounters an error."""


# ---------------------------------------------------------------------------
# Age backend configuration
# ---------------------------------------------------------------------------

_DEFAULT_SECRETS_PATH = Path("~/.config/polysignallab/secrets.age")

# SSH host key types in preference order (strongest first).
_SSH_KEY_TYPES = ("ed25519", "ecdsa", "rsa")
_SSH_KEY_DIR = Path("/etc/ssh")


def detect_ssh_host_key() -> tuple[Path, Path] | None:
    """Find the best available SSH host key pair.

    Tries key types in preference order (ed25519 > ecdsa > rsa).
    Returns ``(public_key_path, private_key_path)`` or None if no
    readable key pair is found.
    """
    for key_type in _SSH_KEY_TYPES:
        pub = _SSH_KEY_DIR / f"ssh_host_{key_type}_key.pub"
        priv = _SSH_KEY_DIR / f"ssh_host_{key_type}_key"
        if not pub.exists():
            log.info("SSH host key probe: %s not found", pub)
            continue
        if not os.access(str(priv), os.R_OK):
            log.warning("SSH host key probe: %s exists but %s not readable", pub, priv)
            continue
        log.info("SSH host key detected: %s (%s)", priv, key_type)
        return pub, priv
    return None


@dataclass(frozen=True, slots=True)
class AgeConfig:
    """Configuration for the age-encrypted secrets backend."""

    secrets_path: Path = field(default=_DEFAULT_SECRETS_PATH)
    ssh_public_key_path: Path | None = field(default=None)
    ssh_private_key_path: Path | None = field(default=None)


_age_config: AgeConfig = AgeConfig()


def configure_age(
    *,
    secrets_path: Path | None = None,
    ssh_public_key_path: Path | None = None,
    ssh_private_key_path: Path | None = None,
) -> None:
    """Override age backend paths.  Call before the first ``get_secret()``."""
    global _age_config, _has_age_cached  # noqa: PLW0603
    _has_age_cached = None  # reset detection cache
    _age_config = AgeConfig(
        secrets_path=secrets_path or _age_config.secrets_path,
        ssh_public_key_path=ssh_public_key_path or _age_config.ssh_public_key_path,
        ssh_private_key_path=ssh_private_key_path or _age_config.ssh_private_key_path,
    )


# ---------------------------------------------------------------------------
# Backend detection and announcement
# ---------------------------------------------------------------------------

_HAS_WINCRED = sys.platform == "win32"

_has_age_cached: bool | None = None


def _resolve_age_keys() -> tuple[Path, Path] | None:
    """Resolve the SSH key pair: explicit config or auto-detect."""
    cfg = _age_config
    if cfg.ssh_public_key_path and cfg.ssh_private_key_path:
        pub = cfg.ssh_public_key_path.expanduser()
        priv = cfg.ssh_private_key_path.expanduser()
        if not pub.exists():
            log.warning("configured SSH public key not found: %s", pub)
            return None
        if not os.access(str(priv), os.R_OK):
            log.warning("configured SSH private key not readable: %s", priv)
            return None
        return pub, priv
    return detect_ssh_host_key()


def _has_age_backend() -> bool:
    """Return True if the age binary and SSH host keys are available."""
    global _has_age_cached  # noqa: PLW0603
    if _has_age_cached is not None:
        return _has_age_cached

    if not shutil.which("age"):
        log.info("age backend: 'age' binary not found in PATH")
        _has_age_cached = False
        return False

    keys = _resolve_age_keys()
    if keys is None:
        log.info(
            "age backend: no readable SSH host key found in %s (tried: %s)",
            _SSH_KEY_DIR,
            ", ".join(_SSH_KEY_TYPES),
        )
        _has_age_cached = False
        return False

    _has_age_cached = True
    return True


_backend_announced = False


def _announce_backend() -> None:
    """Log the credential backend once on first access."""
    global _backend_announced  # noqa: PLW0603  # one-shot flag
    if _backend_announced:
        return
    _backend_announced = True
    if _HAS_WINCRED:
        log.info(
            "credential backend: Windows Credential Manager (DPAPI) — "
            "secrets encrypted at rest, tied to user account"
        )
    elif _has_age_backend():
        keys = _resolve_age_keys()
        key_path = keys[1] if keys else "unknown"
        log.info(
            "credential backend: age-encrypted secrets file (%s) — "
            "secrets encrypted with SSH host key (%s)",
            _age_config.secrets_path.expanduser(),
            key_path,
        )
    else:
        log.info(
            "credential backend: environment variables / .env (legacy) — "
            "no OS credential store available on %s; "
            "install 'age' for encrypted storage",
            sys.platform,
        )


def backend_name() -> str:
    """Return a human-readable name for the active credential backend."""
    if _HAS_WINCRED:
        return "Windows Credential Manager"
    if _has_age_backend():
        return "age-encrypted secrets file"
    return "environment variables (legacy)"


# ---------------------------------------------------------------------------
# Windows Credential Manager via ctypes (no external dependencies)
# ---------------------------------------------------------------------------

if _HAS_WINCRED:
    import ctypes
    import ctypes.wintypes

    _advapi32 = ctypes.windll.advapi32

    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.wintypes.DWORD),
            ("dwHighDateTime", ctypes.wintypes.DWORD),
        ]

    class _CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", ctypes.wintypes.DWORD),
            ("Type", ctypes.wintypes.DWORD),
            ("TargetName", ctypes.c_wchar_p),
            ("Comment", ctypes.c_wchar_p),
            ("LastWritten", _FILETIME),
            ("CredentialBlobSize", ctypes.wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", ctypes.wintypes.DWORD),
            ("AttributeCount", ctypes.wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", ctypes.c_wchar_p),
            ("UserName", ctypes.c_wchar_p),
        ]

    _PCREDENTIAL = ctypes.POINTER(_CREDENTIAL)

    def _wincred_read(target: str) -> str | None:
        """Read a credential blob from Windows Credential Manager."""
        cred_ptr = _PCREDENTIAL()
        ok = _advapi32.CredReadW(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(cred_ptr))
        if not ok:
            return None
        try:
            cred = cred_ptr.contents
            blob_size = cred.CredentialBlobSize
            if blob_size == 0:
                return None
            blob = bytes(cred.CredentialBlob[i] for i in range(blob_size))
            return blob.decode("utf-8").strip()
        finally:
            _advapi32.CredFree(cred_ptr)

    def _wincred_write(target: str, value: str, comment: str = "") -> bool:
        """Write a credential blob to Windows Credential Manager."""
        blob = value.encode("utf-8")
        cred = _CREDENTIAL()
        cred.Flags = 0
        cred.Type = _CRED_TYPE_GENERIC
        cred.TargetName = target
        cred.Comment = comment or target
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = (ctypes.c_ubyte * len(blob))(*blob)
        cred.Persist = _CRED_PERSIST_LOCAL_MACHINE
        cred.AttributeCount = 0
        cred.Attributes = None
        cred.TargetAlias = None
        cred.UserName = "PolySignalLab"
        return bool(_advapi32.CredWriteW(ctypes.byref(cred), 0))

    def _wincred_delete(target: str) -> bool:
        """Delete a credential from Windows Credential Manager."""
        return bool(_advapi32.CredDeleteW(target, _CRED_TYPE_GENERIC, 0))


# ---------------------------------------------------------------------------
# Age-encrypted secrets file backend
# ---------------------------------------------------------------------------


def _age_decrypt(ciphertext_path: Path, identity_path: Path) -> str:
    """Decrypt an age file using the given identity (SSH private key)."""
    try:
        result = subprocess.run(  # noqa: S603  # trusted args
            ["age", "-d", "-i", str(identity_path), str(ciphertext_path)],  # noqa: S607
            capture_output=True,
            check=False,
            timeout=_AGE_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise AgeBackendError("age decrypt timed out") from exc
    except FileNotFoundError as exc:
        raise AgeBackendError("age binary not found") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AgeBackendError(f"age decrypt failed: {stderr}")
    return result.stdout.decode("utf-8")


def _age_encrypt(plaintext: str, recipient_pub_path: Path) -> bytes:
    """Encrypt plaintext using age with the SSH public key as recipient."""
    try:
        result = subprocess.run(  # noqa: S603  # trusted args
            ["age", "-R", str(recipient_pub_path)],  # noqa: S607
            input=plaintext.encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=_AGE_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise AgeBackendError("age encrypt timed out") from exc
    except FileNotFoundError as exc:
        raise AgeBackendError("age binary not found") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AgeBackendError(f"age encrypt failed: {stderr}")
    return result.stdout


_AGE_FORMAT_VERSION = 1


def _age_read_all() -> dict[str, str]:
    """Decrypt and parse the secrets file.  Returns ``{}`` if absent."""
    cfg = _age_config
    secrets_path = cfg.secrets_path.expanduser()
    if not secrets_path.exists():
        log.info("age secrets file not found at %s — no secrets loaded", secrets_path)
        return {}

    keys = _resolve_age_keys()
    if keys is None:
        raise AgeBackendError("no readable SSH host key found for decryption")
    _pub, priv = keys

    plaintext = _age_decrypt(secrets_path, priv)

    try:
        raw = json.loads(plaintext)
    except json.JSONDecodeError as exc:
        raise AgeBackendError(f"secrets file contains invalid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise AgeBackendError("secrets file does not contain a JSON object")

    version = raw.get("_version", 1)
    if version != _AGE_FORMAT_VERSION:
        raise AgeBackendError(
            f"unsupported secrets file version {version} (expected {_AGE_FORMAT_VERSION})"
        )

    # Extract only string-valued entries (metadata like _version is int)
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def _age_write_all(secrets: dict[str, str]) -> None:
    """Encrypt and atomically write the secrets dict."""
    cfg = _age_config
    secrets_path = cfg.secrets_path.expanduser()

    keys = _resolve_age_keys()
    if keys is None:
        raise AgeBackendError("no readable SSH host key found for encryption")
    pub, _priv = keys

    # Ensure parent directory exists with restricted permissions
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        os.chmod(str(secrets_path.parent), 0o700)

    payload: dict[str, str | int] = {
        "_version": _AGE_FORMAT_VERSION,
        "_updated_utc": datetime.now(UTC).isoformat(),
    }
    payload.update(secrets)
    plaintext = json.dumps(payload, indent=2, sort_keys=True)

    ciphertext = _age_encrypt(plaintext, pub)

    # Atomic write: temp file → os.replace()
    fd = -1
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(secrets_path.parent), suffix=".tmp")
        os.write(fd, ciphertext)
        os.close(fd)
        fd = -1
        if sys.platform != "win32":
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, str(secrets_path))
        log.info("age secrets file written: %s", secrets_path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _age_read(name: str) -> str | None:
    """Read a single secret from the age-encrypted file."""
    return _age_read_all().get(name)


def _age_write(name: str, value: str) -> bool:
    """Write a single secret (read-modify-write)."""
    secrets = _age_read_all()
    secrets[name] = value
    _age_write_all(secrets)
    return True


def _age_delete(name: str) -> bool:
    """Remove a single secret from the age-encrypted file."""
    secrets = _age_read_all()
    if name not in secrets:
        return False
    del secrets[name]
    _age_write_all(secrets)
    return True


# ---------------------------------------------------------------------------
# Env-var scrub helper
# ---------------------------------------------------------------------------


def _scrub_env_var(source_label: str, name: str, env_var: str | None) -> None:
    """Remove an env var from the process if a higher-tier backend loaded it."""
    if env_var and env_var in os.environ:
        log.info(
            "credential '%s' loaded from %s; clearing %s from process environment",
            name,
            source_label,
            env_var,
        )
        del os.environ[env_var]


# ---------------------------------------------------------------------------
# Generic secret API (any named credential)
# ---------------------------------------------------------------------------


def get_secret(name: str, *, env_var: str | None = None) -> str | None:
    """Load a named secret from the best available source.

    Lookup order:
      1. Windows Credential Manager (target: ``PolySignalLab/<name>``)
      2. Age-encrypted secrets file
      3. Environment variable ``env_var`` (deprecated legacy fallback)

    When a secret is loaded from a credential store and the same env var
    is set in the process environment, the env var is cleared from the
    process to avoid leaking it to child processes.

    Returns the secret string, or None if not found anywhere.
    """
    _announce_backend()
    target = _CRED_PREFIX + name

    # 1. Try Windows Credential Manager
    if _HAS_WINCRED:
        try:
            val = _wincred_read(target)
            if val:
                log.info("credential '%s' loaded from Credential Manager", name)
                _scrub_env_var("Credential Manager", name, env_var)
                return val
        except OSError as exc:
            log.warning("Credential Manager read failed for '%s': %s", name, exc)

    # 2. Try age-encrypted secrets file
    if _has_age_backend():
        try:
            val = _age_read(name)
            if val:
                log.info("credential '%s' loaded from age secrets file", name)
                _scrub_env_var("age secrets file", name, env_var)
                return val
        except AgeBackendError as exc:
            log.warning("age backend read failed for '%s': %s", name, exc)

    # 3. Try environment variable (deprecated legacy fallback)
    if env_var:
        val = os.environ.get(env_var, "").strip()
        if val:
            log.info(
                "credential '%s' loaded from environment variable %s (legacy)",
                name,
                env_var,
            )
            return val

    log.warning("credential '%s' not found in any source", name)
    return None


def store_secret(name: str, value: str) -> bool:
    """Store a named secret in the best available credential store.

    Tries Windows Credential Manager first, then age-encrypted file.
    Returns False if no credential store is available.
    """
    if _HAS_WINCRED:
        target = _CRED_PREFIX + name
        if _wincred_write(target, value, comment=f"PolySignalLab {name}"):
            log.info("secret '%s' stored in Windows Credential Manager", name)
            return True
        log.warning("failed to write secret '%s' to Credential Manager", name)
        return False

    if _has_age_backend():
        try:
            if _age_write(name, value):
                log.info("secret '%s' stored in age secrets file", name)
                return True
        except AgeBackendError as exc:
            log.warning("age backend write failed for '%s': %s", name, exc)
        return False

    return False


def delete_secret(name: str) -> bool:
    """Remove a named secret from the credential store."""
    if _HAS_WINCRED:
        target = _CRED_PREFIX + name
        if _wincred_delete(target):
            log.info("secret '%s' deleted from Credential Manager", name)
            return True
        return False

    if _has_age_backend():
        try:
            if _age_delete(name):
                log.info("secret '%s' deleted from age secrets file", name)
                return True
        except AgeBackendError as exc:
            log.warning("age backend delete failed for '%s': %s", name, exc)
        return False

    return False


# ---------------------------------------------------------------------------
# HMAC key API (with hex validation)
# ---------------------------------------------------------------------------


def _validate_hex_key(hex_key: str) -> bytes:
    """Validate and decode a hex key string. Raises ValueError on failure."""
    key = bytes.fromhex(hex_key)
    if len(key) < _MIN_HMAC_KEY_BYTES:
        raise ValueError(
            f"HMAC key is too short ({len(key)} bytes) — "
            f"minimum {_MIN_HMAC_KEY_BYTES} bytes ({_MIN_HMAC_KEY_BYTES * 2} hex chars)"
        )
    return key


def get_hmac_key() -> bytes:
    """Load the HMAC pre-shared key from the best available source.

    Lookup order:
      1. Windows Credential Manager (Windows only)
      2. Age-encrypted secrets file
      3. PLSLAB_HMAC_KEY environment variable (deprecated legacy)

    Raises ValueError if no key is found or the key is invalid.
    """
    _announce_backend()

    # 1. Try Windows Credential Manager
    if _HAS_WINCRED:
        try:
            hex_key = _wincred_read(_HMAC_CRED_TARGET)
            if hex_key:
                key = _validate_hex_key(hex_key)
                log.info("HMAC key loaded from Windows Credential Manager")
                _scrub_env_var("Credential Manager", "HMAC_KEY", _HMAC_ENV_KEY)
                return key
        except (ValueError, OSError) as exc:
            log.warning("credential manager key invalid, falling back: %s", exc)

    # 2. Try age-encrypted secrets file
    if _has_age_backend():
        try:
            hex_key = _age_read("HMAC_KEY")
            if hex_key:
                key = _validate_hex_key(hex_key)
                log.info("HMAC key loaded from age secrets file")
                _scrub_env_var("age secrets file", "HMAC_KEY", _HMAC_ENV_KEY)
                return key
        except (ValueError, AgeBackendError) as exc:
            log.warning("age backend HMAC key invalid or read failed: %s", exc)

    # 3. Try environment variable (deprecated legacy)
    raw = os.environ.get(_HMAC_ENV_KEY, "").strip()
    if raw:
        key = _validate_hex_key(raw)
        log.info(
            "HMAC key loaded from %s environment variable (legacy)",
            _HMAC_ENV_KEY,
        )
        return key

    # Nothing found
    sources: list[str] = []
    if _HAS_WINCRED:
        sources.append("Windows Credential Manager")
    if _has_age_backend():
        sources.append("age secrets file")
    sources.append(f"{_HMAC_ENV_KEY} environment variable")
    raise ValueError(
        f"HMAC key not found in {' or '.join(sources)}. Run: python psl.py generate-key"
    )


def store_key(hex_key: str) -> bool:
    """Store the HMAC key in the OS credential store.

    Validates the key before storing.  Raises ValueError if invalid.
    """
    _validate_hex_key(hex_key)  # fail fast on bad input
    return store_secret("HMAC_KEY", hex_key)


def delete_key() -> bool:
    """Remove the HMAC key from the OS credential store."""
    return delete_secret("HMAC_KEY")


def has_credential_store() -> bool:
    """Return True if a credential store is available (WinCred or age)."""
    return _HAS_WINCRED or _has_age_backend()


# ---------------------------------------------------------------------------
# .env residual detection
# ---------------------------------------------------------------------------

_SENSITIVE_ENV_VARS = (
    "POLY_PRIVATE_KEY",
    "POLY_API_KEY",
    "POLY_API_SECRET",
    "POLY_API_PASSPHRASE",
    "POLY_FUNDER_ADDRESS",
    _HMAC_ENV_KEY,
)


def _is_secret_managed(cred_name: str) -> bool:
    """Return True if the given credential exists in a credential store."""
    if _HAS_WINCRED:
        try:
            if _wincred_read(_CRED_PREFIX + cred_name):
                return True
        except OSError as exc:
            log.warning("managed check: WinCred read failed for '%s': %s", cred_name, exc)

    if _has_age_backend():
        try:
            if _age_read(cred_name):
                return True
        except AgeBackendError as exc:
            log.warning("managed check: age read failed for '%s': %s", cred_name, exc)

    return False


def warn_env_file_secrets(env_paths: list[Path]) -> list[str]:
    """Check .env files for secrets that are already in a credential store.

    Returns a list of human-readable warning strings (empty if nothing
    to warn about).  Only runs on platforms with a credential store.
    """
    if not has_credential_store():
        return []

    # Which secrets are actually stored in a credential store?
    managed: list[str] = []
    for env_var in _SENSITIVE_ENV_VARS:
        cred_name = _env_var_to_cred_name(env_var)
        if _is_secret_managed(cred_name):
            managed.append(env_var)

    if not managed:
        return []

    # Scan .env files for those same variables
    warnings: list[str] = []
    for env_path in env_paths:
        if not env_path.exists():
            continue
        try:
            content = env_path.read_text(encoding="utf-8")
        except OSError:
            continue
        found_in_file: list[str] = []
        for env_var in managed:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith(f"{env_var}="):
                    val = stripped.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        found_in_file.append(env_var)
                        break
        if found_in_file:
            store = backend_name()
            warnings.append(
                f"{store} is active but {env_path} still "
                f"contains: {', '.join(found_in_file)}. "
                f"Delete those lines — they are no longer needed on "
                f"this machine."
            )
    return warnings


def _env_var_to_cred_name(env_var: str) -> str:
    """Map an environment variable name to its Credential Manager name."""
    mapping = {
        "POLY_PRIVATE_KEY": "private_key",
        "POLY_API_KEY": "api_key",
        "POLY_API_SECRET": "api_secret",
        "POLY_API_PASSPHRASE": "api_passphrase",
        "POLY_FUNDER_ADDRESS": "funder_address",
        _HMAC_ENV_KEY: "HMAC_KEY",
    }
    return mapping.get(env_var, env_var)
