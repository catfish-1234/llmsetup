"""Register the finished setup with every local LLM app the user uses."""

from llmsetup.interop.export import available_targets, register_everywhere

__all__ = ["available_targets", "register_everywhere"]
