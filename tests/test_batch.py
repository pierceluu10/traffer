import pytest
import os
import tempfile
from PIL import Image
from mlops.batch_inference import run_batch, summarize_batch


@pytest.fixture
def image_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            img = Image.new("RGB", (640, 480), color="red")
            img.save(os.path.join(tmpdir, f"img_{i}.jpg"))
        yield tmpdir


@pytest.fixture
def empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_batch_runs(image_dir):
    results = run_batch(image_dir)
    assert len(results) == 3


def test_batch_result_structure(image_dir):
    results = run_batch(image_dir)
    for r in results:
        assert "file" in r
        assert "detections" in r
        assert "count" in r


def test_batch_output_file(image_dir):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name
    run_batch(image_dir, output_path=output_path)
    assert os.path.exists(output_path)
    os.unlink(output_path)


def test_batch_skips_non_images(image_dir):
    with open(os.path.join(image_dir, "readme.txt"), "w") as f:
        f.write("not an image")
    results = run_batch(image_dir)
    assert len(results) == 3


def test_summarize_batch(image_dir):
    results = run_batch(image_dir)
    summary = summarize_batch(results)
    assert summary["total_images"] == 3
    assert "total_detections" in summary
    assert "avg_detections_per_image" in summary
    assert "label_distribution" in summary


def test_batch_empty_directory(empty_dir):
    results = run_batch(empty_dir)
    assert results == []


def test_summarize_empty_batch():
    summary = summarize_batch([])
    assert summary["total_images"] == 0
    assert summary["total_detections"] == 0
    assert summary["avg_detections_per_image"] == 0


def test_batch_confidence_filtering(image_dir):
    low = run_batch(image_dir, confidence=0.1)
    high = run_batch(image_dir, confidence=0.99)
    low_total = sum(r["count"] for r in low)
    high_total = sum(r["count"] for r in high)
    assert high_total <= low_total
