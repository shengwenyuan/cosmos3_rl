import json
from pathlib import Path

import numpy as np
import pytest

from tools.rh20t.gt_chunk_audit.analyze import AXES, materialize_audit, summarize_chunks


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = np.zeros((2, len(AXES)), dtype=np.float32)
    gt = np.zeros((2, 32, len(AXES)), dtype=np.float32)
    gt[..., 0] = np.linspace(0.01, 0.32, 32)
    raw = gt * 2.0
    post = gt.copy()
    return current, raw, post, gt


def test_summary_compares_axes_and_selected_horizons() -> None:
    summary = summarize_chunks(*_arrays())

    shoulder = summary["axes"]["shoulder_pan_joint"]
    assert summary["queries"] == 2
    assert summary["chunk_size"] == 32
    assert shoulder["raw_motion_scale"] == pytest.approx(2.0)
    assert shoulder["post_motion_scale"] == pytest.approx(1.0)
    assert shoulder["post_mae"] == 0.0
    assert set(shoulder["horizons"]) == {"1", "8", "16", "32"}


def test_materialize_writes_one_lossless_record_per_query(tmp_path: Path) -> None:
    current, raw, post, gt = _arrays()
    samples = tmp_path / "capture_samples.npz"
    report = tmp_path / "capture.json"
    output = tmp_path / "audit"
    np.savez_compressed(
        samples,
        current=current,
        raw_prediction=raw,
        prediction=post,
        target=gt,
    )
    report.write_text(json.dumps({"indices": [41, 99], "checkpoint": "/ckpt"}), encoding="utf-8")

    summary = materialize_audit(samples, report, output)

    assert summary["dataset_indices"] == [41, 99]
    assert (output / "per_axis_horizon.csv").is_file()
    with np.load(output / "query_000_dataset_41" / "chunks.npz") as saved:
        np.testing.assert_array_equal(saved["current"], current[0])
        np.testing.assert_array_equal(saved["raw"], raw[0])
        np.testing.assert_array_equal(saved["post"], post[0])
        np.testing.assert_array_equal(saved["gt"], gt[0])


def test_materialize_refuses_to_overwrite(tmp_path: Path) -> None:
    samples = tmp_path / "capture_samples.npz"
    current, raw, post, gt = _arrays()
    np.savez_compressed(samples, current=current, raw_prediction=raw, prediction=post, target=gt)
    output = tmp_path / "audit"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_audit(samples, None, output)
