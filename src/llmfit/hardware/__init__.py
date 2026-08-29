"""Hardware detection. Best-effort, never fatal, always overridable."""

from llmfit.hardware.detect import apply_overrides, detect_hardware

__all__ = ["apply_overrides", "detect_hardware"]
