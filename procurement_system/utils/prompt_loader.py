"""
utils/prompt_loader.py

Prompt text file loader for the Multi-Agent Procurement System.

Responsibilities:
    - Read a prompt .txt file from disk and return its content as a string
    - Optionally verify the file checksum against a declared value
    - Raise clear custom exceptions if the file is missing or tampered with
    - Stay generic — no knowledge of which agent uses which prompt
      (project_paths.py and settings.py handle those concerns)

Caching:
    NOT applied here — belongs in settings.py (lru_cache getters).
    This keeps load_prompt() pure and easily testable in isolation.

Checksum verification:
    When expected_checksum is provided, the SHA-256 hash of the file
    content is compared against it. This detects accidental or
    unauthorised prompt edits in production.
    Checksums are declared in config/config_*_agent.yaml.
"""

import hashlib
import logging
from pathlib import Path

from procurement_system.exceptions import (
    ConfigFileNotFoundError,
    InvalidConfigError,
)

logger = logging.getLogger(__name__)


def load_prompt(path: str | Path, expected_checksum: str | None = None) -> str:
    """
    Read a prompt text file and return its content as a string.

    Optionally verifies the SHA-256 checksum of the file content
    against a declared value from config_*_agent.yaml.

    Args:
        path:
            Absolute or relative path to the .txt prompt file.
            Pass str(Path) from project_paths.py for safety.
        expected_checksum:
            Optional. SHA-256 checksum in the format "sha256:<hex>".
            If provided and the file hash does not match, raises
            InvalidConfigError. Pass None to skip verification.

    Returns:
        Full prompt text as a string (UTF-8, trailing whitespace stripped).

    Raises:
        ConfigFileNotFoundError: File does not exist at the given path.
        InvalidConfigError:      File exists but is empty, unreadable,
                                 or checksum verification fails.

    Example:
        # Without checksum verification (development)
        from procurement_system.utils.prompt_loader import load_prompt
        from procurement_system.project_paths import INTAKE_AGENT_SYSTEM_PROMPT

        prompt = load_prompt(INTAKE_AGENT_SYSTEM_PROMPT)  # Path or str

        # With checksum verification (production)
        cfg = get_intake_config()
        checksum = cfg["intake_agent"]["prompt"]["checksums"]["system"]
        prompt = load_prompt(INTAKE_AGENT_SYSTEM_PROMPT, checksum)  # Path or str
    """
    logger.debug(f"Loading prompt file: {path}")

    # ── File existence ────────────────────────────────────
    file_path = Path(path)

    if not file_path.exists():
        raise ConfigFileNotFoundError(path=path)

    # ── Read content ──────────────────────────────────────
    try:
        content = file_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise InvalidConfigError(
            file=path,
            reason=f"Could not read file: {e}"
        ) from e

    # ── Empty file ────────────────────────────────────────
    if not content:
        raise InvalidConfigError(
            file=path,
            reason="Prompt file is empty"
        )

    # ── Checksum verification (production safeguard) ──────
    if expected_checksum is not None:
        _verify_checksum(
            content=content,
            expected=expected_checksum,
            path=path,
        )

    logger.debug(
        f"Prompt loaded: {path} "
        f"({len(content)} chars, checksum={'verified' if expected_checksum else 'skipped'})"
    )
    return content


def compute_checksum(path: str | Path) -> str:
    """
    Compute the SHA-256 checksum of a prompt file.

    Use this to generate checksums after editing a prompt file,
    then paste the result into config/config_*_agent.yaml.

    Args:
        path: Path to the .txt prompt file.

    Returns:
        Checksum string in the format "sha256:<hex>".

    Example:
        from procurement_system.utils.prompt_loader import compute_checksum
        from procurement_system.project_paths import INTAKE_AGENT_SYSTEM_PROMPT

        print(compute_checksum(INTAKE_AGENT_SYSTEM_PROMPT))  # Path or str
        # sha256:a3f2c1...
    """
    content = Path(path).read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ── Internal helpers ──────────────────────────────────────

def _verify_checksum(content: str, expected: str, path: str) -> None:
    """
    Verify SHA-256 checksum of prompt content.

    Args:
        content:  File content as string.
        expected: Expected checksum in format "sha256:<hex>".
        path:     File path — used in error message only.

    Raises:
        InvalidConfigError: If checksum does not match or format is invalid.
    """
    if not expected.startswith("sha256:"):
        raise InvalidConfigError(
            file=path,
            reason=(
                f"Invalid checksum format: '{expected}'. "
                f"Expected format: 'sha256:<hex>'"
            )
        )

    expected_hex = expected.removeprefix("sha256:")
    actual_hex   = hashlib.sha256(content.encode("utf-8")).hexdigest()

    if actual_hex != expected_hex:
        raise InvalidConfigError(
            file=path,
            reason=(
                f"Checksum mismatch — prompt file may have been modified.\n"
                f"  Expected: sha256:{expected_hex}\n"
                f"  Actual:   sha256:{actual_hex}\n"
                f"  Update the checksum in config/config_intake_agent.yaml "
                f"after intentional edits."
            )
        )

    logger.debug(f"Checksum verified: {path}")
