"""CLI surface and the pipeline wiring, exercised without a real model."""

from __future__ import annotations

import json

import pytest

from llmsetup.cli import build_parser, main
from llmsetup.hardware.detect import apply_overrides, detect_hardware
from llmsetup.report.render import render_markdown
from llmsetup.types import GIB, Accelerator, Task, UseCaseProfile


@pytest.fixture(autouse=True)
def no_catalog_network(monkeypatch):
    # CLI tests exercise the bundled snapshot; updater tests fake their source.
    monkeypatch.setenv("LLMSETUP_AUTO_UPDATE", "0")


class TestParser:
    def test_every_subcommand_parses(self):
        parser = build_parser()
        for argv in (
            ["detect"], ["plan", "--task", "chat"], ["setup", "--dry-run"],
            ["catalog", "list"], ["catalog", "show", "qwen3-8b"], ["runtimes"],
            ["verify", "--model", "m"], ["export", "--model", "qwen3-8b"],
            ["doctor"],
        ):
            assert parser.parse_args(argv).command == argv[0]

    def test_repeatable_flags_accumulate(self):
        args = build_parser().parse_args(
            ["plan", "--task", "chat", "--task", "coding", "--client", "ollama"])
        assert args.tasks == ["chat", "coding"]
        assert args.clients == ["ollama"]


class TestDetect:
    def test_runs_and_emits_json(self, capsys):
        assert main(["detect", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "os" in payload and "gpus" in payload

    def test_never_raises_on_this_machine(self):
        hardware = detect_hardware()
        assert hardware.os
        assert hardware.ram_bytes > 0


class TestOverrides:
    def test_declaring_a_gpu_creates_one(self):
        hardware = apply_overrides(detect_hardware(probe_gpus=False),
                                   {"vram": 24, "gpu_vendor": "nvidia"})
        assert hardware.gpus and hardware.gpus[0].vram_bytes == 24 * GIB
        assert hardware.accelerator is Accelerator.NVIDIA

    def test_overrides_are_recorded_as_warnings(self):
        """A plan made against asserted rather than measured numbers should say so."""
        hardware = apply_overrides(detect_hardware(probe_gpus=False), {"ram": "64G"})
        assert any("overrid" in w for w in hardware.detection_warnings)

    def test_cpu_only_removes_every_gpu(self):
        hardware = apply_overrides(detect_hardware(probe_gpus=False),
                                   {"vram": 24, "gpu_vendor": "nvidia",
                                    "cpu_only": True})
        assert hardware.gpus == []

    def test_bare_numbers_are_read_as_gigabytes(self):
        hardware = apply_overrides(detect_hardware(probe_gpus=False), {"ram": 32})
        assert hardware.ram_bytes == 32 * GIB


class TestPlanCommand:
    def test_emits_ranked_json(self, capsys):
        code = main(["plan", "--json", "--vram", "24", "--ram", "64",
                     "--gpu-vendor", "nvidia", "--task", "coding", "--yes"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["candidates"]
        scores = [c["score"] for c in payload["candidates"]]
        assert scores == sorted(scores, reverse=True)

    def test_explains_the_runtime_choice(self, capsys):
        main(["plan", "--json", "--vram", "24", "--gpu-vendor", "nvidia",
              "--task", "chat", "--yes"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["runtime_reason"]

    def test_oversized_context_is_shortened_with_an_explanation(self, capsys, monkeypatch):
        from conftest import make_hardware

        monkeypatch.setattr("llmsetup.cli.detect_hardware", lambda **kw: make_hardware())
        code = main(["plan", "--task", "chat", "--vision", "--context", "1000000", "--yes"])
        assert code == 0
        assert "automatically using" in capsys.readouterr().err


class TestCatalogCommand:
    def test_lists_models(self, capsys):
        assert main(["catalog", "list"]) == 0
        assert "qwen3-8b" in capsys.readouterr().err

    def test_shows_one_model_as_json(self, capsys):
        assert main(["catalog", "show", "qwen3-8b"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["id"] == "qwen3-8b"

    def test_an_unknown_model_exits_nonzero(self, capsys):
        assert main(["catalog", "show", "not-a-model"]) == 1


class TestRuntimesCommand:
    def test_reports_availability_and_install_hints(self, capsys):
        assert main(["runtimes", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert {"llamacpp", "ollama", "vllm", "mlx", "lmstudio"} <= {r["name"] for r in rows}
        for row in rows:
            if not row["available"]:
                assert row["install"], row["name"]


class TestDryRunPipeline:
    def test_plans_and_tunes_without_touching_the_network(self, capsys, catalog):
        """The whole decision half of the tool runs offline; only provisioning
        needs the network."""
        from conftest import make_hardware

        from llmsetup.pipeline import Pipeline
        from llmsetup.util.log import Console

        pipeline = Pipeline(
            UseCaseProfile(tasks=[Task.CODING], needs_tools=True, context_tokens=32768),
            make_hardware(vram_gb=24, ram_gb=64), catalog,
            Console(quiet=True), dry_run=True,
        )
        result = pipeline.run()
        assert result.plan.best is not None
        assert result.config.context_tokens > 0
        assert result.config.gpu_layers > 0
        assert result.checks == []          # nothing verified, nothing claimed

    def test_the_report_never_claims_untested_success(self, catalog):
        """A dry run has proved nothing, and the report must reflect that."""
        from conftest import make_hardware

        from llmsetup.pipeline import Pipeline
        from llmsetup.util.log import Console

        pipeline = Pipeline(UseCaseProfile(tasks=[Task.CHAT]),
                            make_hardware(), catalog, Console(quiet=True),
                            dry_run=True)
        markdown = render_markdown(pipeline.run())
        assert "No checks were run" in markdown

    def test_the_markdown_report_records_the_decision(self, catalog):
        from conftest import make_hardware

        from llmsetup.pipeline import Pipeline
        from llmsetup.util.log import Console

        pipeline = Pipeline(
            UseCaseProfile(tasks=[Task.CODING], context_tokens=32768),
            make_hardware(vram_gb=24), catalog, Console(quiet=True), dry_run=True)
        markdown = render_markdown(pipeline.run())
        for expected in ("## What was chosen", "Quantization", "KV cache",
                         "## Machine", "## Verification", "Runtime chosen because"):
            assert expected in markdown


class TestSetupResultSerialisation:
    def test_round_trips_through_json(self, catalog):
        from conftest import make_hardware

        from llmsetup.pipeline import Pipeline
        from llmsetup.util.log import Console

        pipeline = Pipeline(UseCaseProfile(tasks=[Task.CHAT]), make_hardware(),
                            catalog, Console(quiet=True), dry_run=True)
        payload = json.loads(json.dumps(pipeline.run().to_dict(), default=str))
        assert payload["model"]["id"]
        assert payload["config"]["context_tokens"]
        assert "succeeded" in payload


class TestNeverClaimsUntestedSuccess:
    """The tool's central promise: no claim without a check backing it."""

    def _dry_run(self, catalog):
        from conftest import make_hardware

        from llmsetup.pipeline import Pipeline
        from llmsetup.util.log import Console

        return Pipeline(UseCaseProfile(tasks=[Task.CHAT]), make_hardware(),
                        catalog, Console(quiet=True), dry_run=True).run()

    def test_an_untested_plan_is_not_a_success(self, catalog):
        result = self._dry_run(catalog)
        assert result.checks_ran is False
        assert result.succeeded is False

    def test_the_terminal_report_says_nothing_was_tested(self, catalog, capsys):
        import sys

        from llmsetup.report.render import render_terminal
        from llmsetup.util.log import Console

        render_terminal(self._dry_run(catalog), Console(stream=sys.stdout, color=False))
        out = capsys.readouterr().out
        assert "nothing was tested" in out.lower()
        assert "verified" not in out.lower().split("nothing was tested")[0]

    def test_the_markdown_report_disclaims_verification(self, catalog):
        markdown = render_markdown(self._dry_run(catalog))
        assert "Verified: no" in markdown
        assert "no claim is made that it works" in markdown

    def test_skipped_checks_do_not_count_as_having_run(self, catalog):
        from llmsetup.types import CheckResult, CheckStatus, Severity

        result = self._dry_run(catalog)
        result.checks = [CheckResult("context", CheckStatus.SKIP, Severity.ADVISORY,
                                     "skipped")]
        assert result.checks_ran is False
        assert result.succeeded is False

    def test_a_dry_run_still_exits_zero(self, capsys):
        """It proved nothing, but it also failed at nothing."""
        code = main(["setup", "--dry-run", "--vram", "24", "--gpu-vendor", "nvidia",
                     "--task", "chat", "--yes"])
        assert code == 0


class TestReportCommand:
    """A bug report is only actionable if the reporter can produce one easily,
    so every run saves the report whether or not it was asked for."""

    def test_a_setup_saves_its_report_without_being_asked(self, capsys):
        main(["setup", "--dry-run", "--vram", "24", "--gpu-vendor", "nvidia",
              "--task", "chat", "--yes"])
        capsys.readouterr()

        assert main(["report"]) == 0
        printed = capsys.readouterr().out
        assert printed.startswith("# Local LLM setup:")
        assert "## What was chosen" in printed

    def test_the_reprint_is_the_report_not_a_summary_of_it(self, capsys):
        """It has to carry the decisions and their reasons, which is the part
        that makes a wrong plan diagnosable."""
        main(["setup", "--dry-run", "--vram", "24", "--gpu-vendor", "nvidia",
              "--task", "coding", "--yes"])
        capsys.readouterr()

        main(["report"])
        printed = capsys.readouterr().out
        assert "## Machine" in printed
        assert "Runtime chosen because:" in printed

    def test_says_what_to_do_when_there_is_nothing_saved(self, capsys):
        code = main(["report"])
        assert code != 0
        assert "llmsetup setup" in capsys.readouterr().err

    def test_prints_a_report_it_is_handed(self, capsys, tmp_path):
        saved = tmp_path / "old-report.md"
        saved.write_text("# Local LLM setup: something else\n", encoding="utf-8")
        assert main(["report", str(saved)]) == 0
        assert "something else" in capsys.readouterr().out
