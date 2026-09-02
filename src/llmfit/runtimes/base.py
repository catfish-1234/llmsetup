"""The runtime abstraction.

Every inference engine llmfit supports is reduced to the same five questions:

    is it here?  can it run this model?  how do I get the weights?
    how do I start it?  where do I talk to it?

Keeping the interface this small is what lets the verifier and repair loop be
engine-agnostic: they operate on an :class:`Endpoint`, never on Ollama or
llama.cpp specifics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmfit.types import ModelSpec, RuntimeConfig
from llmfit.util.log import Console


@dataclass
class Endpoint:
    """A running, OpenAI-compatible inference server."""

    base_url: str
    model: str
    runtime: str
    api_style: str = "openai"      # openai | ollama
    headers: Dict[str, str] = field(default_factory=dict)
    supports_embeddings: bool = False
    started_by_us: bool = False
    log_path: Optional[str] = None
    process: Any = None            # subprocess.Popen when we launched it

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/models"

    @property
    def embeddings_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/embeddings"

    @property
    def infill_url(self) -> str:
        """Fill-in-the-middle. Not an OpenAI route, so it sits outside /v1."""
        return f"{self.base_url.rstrip('/')}/infill"


class RuntimeUnavailable(Exception):
    """Raised by :meth:`Runtime.provision` when the engine cannot be used."""

    def __init__(self, message: str, install_hint: str = ""):
        super().__init__(message)
        self.install_hint = install_hint


class Runtime(abc.ABC):
    """Base class for inference engine adapters."""

    name: str = "abstract"
    #: Human label used in reports.
    label: str = "Abstract runtime"
    #: Whether this engine consumes GGUF files (vs. safetensors).
    uses_gguf: bool = True

    # ------------------------------------------------------------- capability

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Is the engine installed and launchable on this machine?"""

    def install_hint(self) -> str:
        """What the user should run to get it. Shown when unavailable."""
        return ""

    def version(self) -> Optional[str]:
        return None

    def supports(self, spec: ModelSpec) -> bool:
        """Can this engine serve this model at all?"""
        if self.uses_gguf:
            return bool(spec.gguf_repo or spec.ollama_name)
        return bool(spec.hf_repo)

    # ------------------------------------------------------------- lifecycle

    @abc.abstractmethod
    def ensure_model(
        self, spec: ModelSpec, config: RuntimeConfig, console: Console
    ) -> RuntimeConfig:
        """Make the weights available locally. May mutate ``config.model_path``."""

    @abc.abstractmethod
    def start(self, spec: ModelSpec, config: RuntimeConfig, console: Console) -> Endpoint:
        """Launch (or attach to) a server and return a ready endpoint."""

    def stop(self, endpoint: Endpoint) -> None:
        """Shut down a server we started. Attached servers are left alone."""
        if endpoint.started_by_us and endpoint.process is not None:
            proc = endpoint.process
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # ------------------------------------------------------------- reporting

    @abc.abstractmethod
    def launch_command(self, spec: ModelSpec, config: RuntimeConfig) -> List[str]:
        """The command a user could run by hand to reproduce this setup.

        Printed in the final report: the tool should leave people able to run
        their model without it.
        """

    def config_files(self, spec: ModelSpec, config: RuntimeConfig) -> Dict[str, str]:
        """Extra files to write out (Modelfile, YAML), as {relative path: contents}."""
        return {}

    def local_model_path(self, spec: ModelSpec, config: RuntimeConfig) -> Optional[Path]:
        """Where the weights ended up, for the interop layer to link against."""
        return Path(config.model_path) if config.model_path else None
