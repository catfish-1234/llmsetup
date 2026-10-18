"""Retaining a model changes optimization, never the user's requirements."""

from dataclasses import replace

import pytest
from conftest import make_hardware

from llmsetup.catalog.catalog import Catalog
from llmsetup.planner.model_fit import assess_requested_model, resolve_requested_model
from llmsetup.planner.plan import build_adaptive_plan, build_plan
from llmsetup.types import GIB, Accelerator, ModelSpec, Task, UseCaseProfile
from llmsetup.util.errors import CatalogError, PlanningError


def spec(model_id="chosen", params=16, **kwargs):
    values = {"id": model_id, "name": model_id.title(), "publisher": "test", "params_b": params,
              "layers": 16, "hidden_size": 1024, "kv_heads": 4, "max_context": 32768,
              "tasks": ["chat"], "gguf_repo": f"test/{model_id}-GGUF",
              "hf_repo": f"test/{model_id}", "recommended_quants": ["Q4_K_M", "Q8_0"]}
    values.update(kwargs)
    return ModelSpec(**values)


def catalog_of(*models):
    return Catalog(list(models), {"Q4_K_M": 4.85, "Q8_0": 8.5})


def hardware(**kwargs):
    return replace(make_hardware(**kwargs), runtime_availability={"llamacpp": True})


def test_requested_model_round_trips_and_old_profiles_remain_compatible():
    profile = UseCaseProfile(requested_model="chosen")
    assert UseCaseProfile.from_dict(profile.to_dict()).requested_model == "chosen"
    assert UseCaseProfile.from_dict({"tasks": ["chat"]}).requested_model is None


def test_direct_planning_normalizes_a_requested_display_name():
    model = spec(params=3, name="Chosen Model")
    plan = build_plan(UseCaseProfile(requested_model="chosen model"), hardware(),
                      catalog_of(model))
    assert plan.best.model.id == "chosen"
    assert plan.profile.requested_model == "chosen"


def test_requested_model_wins_even_when_a_faster_model_fits_better():
    catalog = catalog_of(spec(), spec("alternative", params=3))
    hw = hardware(vram_gb=8, ram_gb=64)
    plan = build_plan(UseCaseProfile(requested_model="chosen"), hw, catalog)
    assert plan.best.model.id == "chosen"
    assert plan.best.gpu_fraction < 1
    assert plan.fallbacks[0].model.id == "alternative"
    assert plan.fallbacks[0].fits_on_gpu


def test_requested_quantization_is_tuned_to_memory():
    catalog = catalog_of(spec(params=3))
    plan = build_plan(UseCaseProfile(requested_model="chosen", context_tokens=512),
                      hardware(vram_gb=0, ram_gb=3), catalog)
    assert plan.best.quant == "Q4_K_M"
    assert plan.best.estimate.total <= plan.hardware.usable_cpu_bytes


def test_requested_context_can_shrink_without_switching_model():
    catalog = catalog_of(spec(max_context=2048), spec("alternative", params=3))
    plan = build_adaptive_plan(UseCaseProfile(requested_model="chosen", context_tokens=8192),
                               hardware(vram_gb=24), catalog)
    assert plan.best.model.id == "chosen"
    assert plan.best.estimate.context_tokens == 2048
    assert plan.profile.context_tokens == 2048
    assert any("requested 8192" in note for note in plan.notes)


@pytest.mark.parametrize("requirement", [
    {"needs_tools": True}, {"needs_vision": True}, {"needs_json": True},
    {"needs_reasoning": True}, {"commercial_use": True},
])
def test_retaining_model_does_not_override_capability_or_licence(requirement):
    catalog = catalog_of(spec(commercial_ok=False), spec("alternative", params=3,
                         commercial_ok=True, supports_tools=True, supports_vision=True,
                         supports_json=True, is_reasoning=True))
    with pytest.raises(PlanningError):
        build_adaptive_plan(UseCaseProfile(requested_model="chosen", **requirement),
                            hardware(), catalog)


def test_soft_task_mismatch_is_allowed_but_reported():
    catalog = catalog_of(spec(params=3), spec("coder", params=3, tasks=["coding"]))
    profile = UseCaseProfile(requested_model="chosen", tasks=[Task.CODING])
    assessment = assess_requested_model(profile, hardware(), catalog)
    assert assessment.plan.best.model.id == "chosen"
    assert assessment.poor_fit
    assert any("quality for those tasks is uncertain" in text for text in assessment.messages)
    assert assessment.alternatives[0].model.id == "coder"


def test_inline_completion_still_requires_fill_in_the_middle():
    catalog = catalog_of(spec(params=3, tasks=["autocomplete"]))
    with pytest.raises(PlanningError, match="fill-in-the-middle"):
        build_plan(UseCaseProfile(requested_model="chosen", tasks=[Task.AUTOCOMPLETE]),
                   hardware(), catalog)


def test_cpu_is_available_when_almost_nothing_fits_on_detected_gpu():
    catalog = catalog_of(spec(params=32, recommended_quants=["Q4_K_M"]))
    plan = build_plan(UseCaseProfile(requested_model="chosen"),
                      hardware(vram_gb=1, ram_gb=64), catalog)
    assert plan.best.model.id == "chosen"
    assert plan.best.gpu_fraction == 0
    assert plan.best.estimate.total <= plan.hardware.usable_cpu_bytes


def test_shared_apple_memory_is_not_counted_twice_for_partial_offload():
    catalog = catalog_of(spec(params=30, recommended_quants=["Q4_K_M"]))
    hw = hardware(vram_gb=16, ram_gb=16, vendor=Accelerator.APPLE,
                  unified=True, os_name="darwin")
    with pytest.raises(PlanningError):
        build_adaptive_plan(UseCaseProfile(requested_model="chosen"), hw, catalog)


def test_automatic_runtime_uses_gguf_when_native_weights_do_not_fit():
    catalog = catalog_of(spec(params=10))
    hw = replace(hardware(vram_gb=8), runtime_availability={"vllm": True, "llamacpp": True})
    plan = build_plan(UseCaseProfile(requested_model="chosen"), hw, catalog)
    assert plan.runtime == "llamacpp"
    assert plan.best.quant == "Q4_K_M"
    assert plan.best.fits_on_gpu


def test_explicit_native_runtime_never_budgets_gguf_quantization():
    catalog = catalog_of(spec(params=10))
    with pytest.raises(PlanningError):
        build_plan(UseCaseProfile(requested_model="chosen"),
                   hardware(vram_gb=8, ram_gb=128), catalog, runtime="vllm")
    plan = build_plan(UseCaseProfile(requested_model="chosen"),
                      hardware(vram_gb=48), catalog, runtime="vllm")
    assert plan.best.quant == "F16"
    assert plan.best.estimate.weights == int(10 * 1e9 * 16 / 8)


def test_native_fallbacks_use_their_actual_weight_precision():
    catalog = catalog_of(spec(params=2), spec("too-large", params=10))
    plan = build_plan(UseCaseProfile(requested_model="chosen"),
                      hardware(vram_gb=8), catalog, runtime="vllm")
    assert plan.best.quant == "F16"
    assert not plan.fallbacks


def test_native_fallbacks_are_limited_after_all_weight_formats_are_checked():
    chosen = spec(params=1, quality_tier=1)
    oversized = [spec(f"large-{index}", params=10, quality_tier=5)
                 for index in range(8)]
    small = spec("small", params=0.6, quality_tier=1)
    catalog = catalog_of(chosen, *oversized, small)
    hw = hardware(vram_gb=8)
    general = build_plan(UseCaseProfile(), hw, catalog, runtime="vllm")
    assert len(general.candidates) == 8
    assert all(candidate.model.id.startswith("large-") for candidate in general.candidates)
    plan = build_plan(UseCaseProfile(requested_model="chosen"), hw, catalog, runtime="vllm")
    assert [candidate.model.id for candidate in plan.fallbacks] == ["small"]
    assert plan.fallbacks[0].quant == "F16"


def test_native_assessment_finds_small_fit_beyond_general_candidate_limit():
    chosen = spec(params=1, quality_tier=1)
    oversized = [spec(f"large-{index}", params=10, quality_tier=5,
                      tasks=["chat", "coding"]) for index in range(8)]
    small = spec("small", params=0.6, quality_tier=1, tasks=["chat", "coding"],
                 max_context=2048)
    catalog = catalog_of(chosen, *oversized, small)
    profile = UseCaseProfile(requested_model="chosen", tasks=[Task.CHAT, Task.CODING])
    assessment = assess_requested_model(profile, hardware(vram_gb=8), catalog, runtime="vllm")
    assert assessment.poor_fit
    assert [candidate.model.id for candidate in assessment.alternatives] == ["small"]
    assert assessment.alternatives[0].quant == "F16"
    assert assessment.alternatives[0].estimate.context_tokens == 2048
    assert assessment.alternative_runtimes["small"] == "vllm"
    assert any("small" in line and "2,048 tokens" in line for line in assessment.messages)


def test_vllm_planning_reserves_the_runtime_memory_margin():
    catalog = catalog_of(spec(params=3.4))
    with pytest.raises(PlanningError):
        build_plan(UseCaseProfile(requested_model="chosen", context_tokens=512),
                   hardware(vram_gb=8), catalog, runtime="vllm")


def test_mlx_repo_precision_is_preserved_for_later_memory_recomputation():
    catalog = catalog_of(spec(params=3, mlx_repo="test/chosen-8bit"))
    hw = hardware(vram_gb=24, ram_gb=24, unified=True, vendor=Accelerator.APPLE,
                  os_name="darwin")
    plan = build_plan(UseCaseProfile(requested_model="chosen"), hw, catalog, runtime="mlx")
    assert plan.best.quant == "MLX8"
    assert plan.best.estimate.kv_quant == "f16"
    assert any("does not enforce a server context limit" in text for text in plan.best.warnings)
    assert catalog.bpw(plan.best.quant) == 8.5
    assert plan.best.estimate.weights == int(3 * 1e9 * catalog.bpw(plan.best.quant) / 8)


def test_unknown_requested_model_does_not_fall_back_silently():
    with pytest.raises(CatalogError, match="unknown model"):
        build_adaptive_plan(UseCaseProfile(requested_model="missing"), hardware(),
                            catalog_of(spec()))


@pytest.mark.parametrize("alias", ["CHOSEN", "Chosen model", "chosen:16b", "test/chosen-GGUF"])
def test_model_names_tags_and_sources_resolve_without_case_sensitivity(alias):
    model = spec(name="Chosen model", ollama_name="chosen:16b")
    assert resolve_requested_model(alias, catalog_of(model)).id == "chosen"


def test_ambiguous_names_and_partial_names_do_not_guess_a_model():
    catalog = catalog_of(spec("a", name="Shared"), spec("b", name="Shared"))
    with pytest.raises(CatalogError, match="more than one"):
        resolve_requested_model("shared", catalog)
    with pytest.raises(CatalogError, match="unknown model"):
        resolve_requested_model("shar", catalog)


def test_assessment_explains_hardware_quant_context_speed_tasks_and_alternatives():
    catalog = catalog_of(spec(), spec("alternative", params=3))
    assessment = assess_requested_model(UseCaseProfile(requested_model="chosen"),
                                        hardware(vram_gb=8, ram_gb=64), catalog)
    text = " ".join(assessment.messages)
    assert assessment.poor_fit
    assert assessment.plan.best.model.id == "chosen"
    for phrase in ("system RAM", "GPU", "Q4_K_M", "context", "tokens/s", "chat", "Alternative"):
        assert phrase in text
    assert "Estimated" in text
    assert assessment.alternatives[0].model.id == "alternative"
    assert assessment.alternatives[0].estimate.total < 8 * GIB


def test_impossible_model_is_assessed_with_feasible_alternatives():
    catalog = catalog_of(spec(params=235), spec("alternative", params=3))
    assessment = assess_requested_model(UseCaseProfile(requested_model="chosen"),
                                        hardware(vram_gb=8, ram_gb=16), catalog)
    assert assessment.plan is None
    assert assessment.poor_fit
    assert assessment.alternatives[0].model.id == "alternative"
    assert any("does not fit" in line for line in assessment.messages)


def test_alternative_runtime_is_optimized_for_its_actual_weight_format():
    catalog = catalog_of(spec(params=20, gguf_repo=None), spec("alternative", params=5))
    hw = replace(hardware(vram_gb=8, ram_gb=16),
                 runtime_availability={"vllm": True, "llamacpp": False})
    assessment = assess_requested_model(UseCaseProfile(requested_model="chosen"), hw, catalog)
    assert assessment.plan is None
    assert assessment.alternatives[0].model.id == "alternative"
    assert assessment.alternatives[0].quant in {"Q4_K_M", "Q8_0"}
    assert assessment.alternatives[0].estimate.total <= hw.memory_budget
    assert assessment.alternative_runtimes["alternative"] == "llamacpp"


def test_good_fit_needs_no_alternatives():
    assessment = assess_requested_model(UseCaseProfile(requested_model="chosen"),
                                        hardware(vram_gb=12), catalog_of(spec(params=3)))
    assert assessment.plan.best.fits_on_gpu
    assert not assessment.poor_fit
    assert not assessment.alternatives
