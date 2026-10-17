"""A repaired quantization must refer to weights for that quantization."""

from conftest import make_hardware

from llmsetup.pipeline import Pipeline
from llmsetup.types import UseCaseProfile
from llmsetup.util.log import Console


class RecordingRuntime:
    uses_gguf = False

    def __init__(self):
        self.provisioned = []

    def is_available(self):
        return True

    def ensure_model(self, spec, config, console):
        self.provisioned.append(config.quant)
        config.model_path = f"/synthetic/{spec.id}/{config.quant}.gguf"
        return config


def pipeline(catalog, monkeypatch):
    runtime = RecordingRuntime()
    launched = []
    monkeypatch.setattr("llmsetup.pipeline.get_runtime", lambda name: runtime)
    instance = Pipeline(UseCaseProfile(), make_hardware(),
                        catalog.restrict_to("qwen3-8b"), Console(quiet=True),
                        runtime_name="fake", register=False)

    def launch(spec, config, names):
        launched.append((config.quant, config.model_path))
        return []

    monkeypatch.setattr(instance, "_launch_and_check", launch)
    result = instance.run()
    return instance, result.config, runtime, launched


def test_first_quantization_repair_provisions_matching_weights(catalog, monkeypatch):
    instance, config, runtime, launched = pipeline(catalog, monkeypatch)
    original = config.quant
    config.quant = "Q2_K" if original != "Q2_K" else "Q4_K_M"
    instance._relaunch(config, None)
    assert runtime.provisioned == [original, config.quant]
    assert launched[-1] == (config.quant, f"/synthetic/{config.model_id}/{config.quant}.gguf")


def test_setting_only_repair_reuses_the_provisioned_weights(catalog, monkeypatch):
    instance, config, runtime, launched = pipeline(catalog, monkeypatch)
    config.ubatch_size = max(1, config.ubatch_size // 2)
    instance._relaunch(config, None)
    assert runtime.provisioned == [config.quant]
    assert len(launched) == 2
