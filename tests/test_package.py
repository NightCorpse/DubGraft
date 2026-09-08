from importlib.metadata import version

from dubgraft import __version__


def test_installed_package_uses_runtime_version() -> None:
    assert version("dubgraft") == __version__
