"""Loader hints identify known distro packages without executing an installer."""

from pathlib import Path

import pytest

from llmsetup.runtimes.dependencies import shared_library_hint

LOADER_ERROR = (
    "llama-server: error while loading shared libraries: libgomp.so.1: "
    "cannot open shared object file: No such file or directory"
)


@pytest.mark.parametrize("release,command", [
    ("ID=ubuntu\nID_LIKE=debian\n", "apt install libgomp1"),
    ('ID="debian"\n', "apt install libgomp1"),
    ('ID=linuxmint\nID_LIKE="ubuntu debian"\n', "apt install libgomp1"),
    ("ID=fedora\n", "dnf install libgomp"),
    ('ID=rocky\nID_LIKE="rhel centos fedora"\n', "dnf install libgomp"),
    ("ID=arch\n", "pacman -S gcc-libs"),
    ('ID=opensuse-tumbleweed\nID_LIKE="opensuse suse"\n', "zypper install libgomp1"),
])
def test_openmp_hint_uses_distro_and_family(monkeypatch, release, command):
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: release)
    hint = shared_library_hint(LOADER_ERROR)
    assert command in hint
    assert "libgomp.so.1" in hint
    assert "llmsetup setup" in hint


def test_os_release_vendor_location_is_used_if_etc_is_missing(monkeypatch):
    def read_text(path, **kwargs):
        if str(path) == "/etc/os-release":
            raise FileNotFoundError(path)
        assert str(path) == "/usr/lib/os-release"
        return "ID=debian\n"

    monkeypatch.setattr(Path, "read_text", read_text)
    assert "apt install libgomp1" in shared_library_hint(LOADER_ERROR)


@pytest.mark.parametrize("release", ["ID=unknown\n", "", None])
def test_unknown_or_unreadable_distro_does_not_guess_a_package(monkeypatch, release):
    def read_text(*args, **kwargs):
        if release is None:
            raise OSError("cannot read os-release")
        return release

    monkeypatch.setattr(Path, "read_text", read_text)
    hint = shared_library_hint(LOADER_ERROR)
    assert "libgomp.so.1" in hint
    assert "system package" in hint
    assert "apt install" not in hint
    assert "dnf install" not in hint


def test_unknown_library_gets_guidance_without_an_invented_package(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: "ID=ubuntu\n")
    hint = shared_library_hint(LOADER_ERROR.replace("libgomp.so.1", "libexample.so.42"))
    assert "libexample.so.42" in hint
    assert "system package" in hint
    assert "libgomp1" not in hint


@pytest.mark.parametrize("output", [
    "CUDA out of memory", "unsupported CPU", "libgomp.so.1 is installed", "",
])
def test_non_loader_failure_does_not_get_dependency_hint(output):
    assert shared_library_hint(output) == ""
