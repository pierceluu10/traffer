import pytest
import os
import tempfile
from mlops import versioning


@pytest.fixture(autouse=True)
def clean_versions():
    original_dir = versioning.VERSIONS_DIR
    original_file = versioning.REGISTRY_FILE
    with tempfile.TemporaryDirectory() as tmpdir:
        versioning.VERSIONS_DIR = tmpdir
        versioning.REGISTRY_FILE = os.path.join(tmpdir, "registry.json")
        yield
        versioning.VERSIONS_DIR = original_dir
        versioning.REGISTRY_FILE = original_file


@pytest.fixture
def dummy_weights():
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        f.write(b"fake weights")
        yield f.name
    os.unlink(f.name)


def test_register_model(dummy_weights):
    entry = versioning.register_model("1.0.0", dummy_weights, {"accuracy": 0.95})
    assert entry["version"] == "1.0.0"
    assert versioning.get_current_version() == "1.0.0"


def test_list_versions(dummy_weights):
    versioning.register_model("1.0.0", dummy_weights, {"accuracy": 0.90})
    versioning.register_model("1.1.0", dummy_weights, {"accuracy": 0.95})
    versions = versioning.list_versions()
    assert len(versions) == 2


def test_rollback(dummy_weights):
    versioning.register_model("1.0.0", dummy_weights, {"accuracy": 0.90})
    versioning.register_model("2.0.0", dummy_weights, {"accuracy": 0.85})
    assert versioning.get_current_version() == "2.0.0"
    versioning.rollback("1.0.0")
    assert versioning.get_current_version() == "1.0.0"


def test_rollback_nonexistent():
    result = versioning.rollback("99.0.0")
    assert result is None


def test_register_with_notes(dummy_weights):
    entry = versioning.register_model("1.0.0", dummy_weights, {"accuracy": 0.95}, notes="initial release")
    assert entry["notes"] == "initial release"


def test_current_version_empty():
    assert versioning.get_current_version() is None


def test_register_multiple_versions_order(dummy_weights):
    versioning.register_model("1.0.0", dummy_weights, {"accuracy": 0.90})
    versioning.register_model("1.1.0", dummy_weights, {"accuracy": 0.92})
    versioning.register_model("2.0.0", dummy_weights, {"accuracy": 0.95})
    versions = versioning.list_versions()
    assert [v["version"] for v in versions] == ["1.0.0", "1.1.0", "2.0.0"]


def test_rollback_preserves_history(dummy_weights):
    versioning.register_model("1.0.0", dummy_weights, {"accuracy": 0.90})
    versioning.register_model("2.0.0", dummy_weights, {"accuracy": 0.85})
    versioning.rollback("1.0.0")
    versions = versioning.list_versions()
    assert len(versions) == 2  # Both entries still exist


def test_list_versions_empty():
    versions = versioning.list_versions()
    assert versions == []
