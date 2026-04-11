"""Project exception hierarchy (HRP 2.3).

One base exception for all PolySignalLab application errors.
Callers can catch ``AppError`` to handle all project errors without
accidentally swallowing stdlib errors like ``KeyboardInterrupt``.
"""

from __future__ import annotations


class AppError(Exception):
    """Base for all PolySignalLab application errors."""


class ConfigError(AppError):
    """Invalid or missing configuration."""


class ValidationError(AppError):
    """Data failed boundary validation (network, IPC, file input)."""


class ExecutionError(AppError):
    """Order placement, cancellation, or fill-check failure."""


class DataQualityError(AppError):
    """Market data integrity violation (stale feeds, missing fields)."""


class ResolutionError(AppError):
    """Market resolution polling or processing failure."""


class SignalError(AppError):
    """Signal loading, parsing, or lifecycle error."""
