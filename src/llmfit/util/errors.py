"""Exception hierarchy. All user-facing failures carry a remediation hint."""

from __future__ import annotations

from typing import Optional


class LlmfitError(Exception):
    """Base class. ``hint`` is printed to the user as the suggested next step."""

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  hint: {self.hint}" if self.hint else self.message


class DetectionError(LlmfitError):
    """Hardware could not be described well enough to plan against."""


class CatalogError(LlmfitError):
    """The model catalog is missing, malformed, or has no viable entry."""


class PlanningError(LlmfitError):
    """No model fits the constraints."""


class RuntimeNotAvailable(LlmfitError):
    """The chosen inference engine is not installed or not launchable."""


class ProvisionError(LlmfitError):
    """Downloading or launching the model failed."""


class VerificationError(LlmfitError):
    """The verification harness itself could not run."""


class RepairExhausted(LlmfitError):
    """Every applicable remediation was tried and checks still fail."""
