"""
Integration tests for CLI execution.
"""

from pathlib import Path
import subprocess
import sys
import pytest


def test_cli_random_run(tmp_path: Path):
    """Test fast random CLI run with CSV and plot outputs."""
    csv_file = tmp_path / "test_results.csv"
    plot_file = tmp_path / "test_plot.png"
    meta_file = tmp_path / "test_meta.json"

    cmd = [
        sys.executable,
        "-m",
        "network_rheology.cli",
        "random",
        "--nodes", "50",
        "--edges-start", "100",
        "--edges-stop", "200",
        "--edges-step", "50",
        "--dev-start", "1",
        "--dev-stop", "3",
        "--seeds", "5",
        "--save-csv", str(csv_file),
        "--save-plot", str(plot_file),
        "--save-meta", str(meta_file),
        "--no-show",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI execution failed:\n{res.stderr}"

    assert csv_file.exists()
    assert plot_file.exists()
    assert meta_file.exists()


def test_cli_brick3d_run(tmp_path: Path):
    """Test brick3d CLI run."""
    csv_file = tmp_path / "test_brick.csv"
    plot_file = tmp_path / "test_brick.png"

    cmd = [
        sys.executable,
        "-m",
        "network_rheology.cli",
        "brick3d",
        "--layers", "2",
        "--depths", "1", "2",
        "--seeds", "2",
        "--save-csv", str(csv_file),
        "--save-plot", str(plot_file),
        "--no-show",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI brick3d failed:\n{res.stderr}"

    assert csv_file.exists()
    assert plot_file.exists()


def test_legacy_script_run(tmp_path: Path):
    """Test that random_resistor_network.py runs successfully."""
    csv_file = tmp_path / "legacy_results.csv"

    cmd = [
        sys.executable,
        "random_resistor_network.py",
        "--nodes", "50",
        "--edges-start", "100",
        "--edges-stop", "150",
        "--edges-step", "50",
        "--dev-start", "1",
        "--dev-stop", "2",
        "--seeds", "5",
        "--save-csv", str(csv_file),
        "--no-show",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Legacy script failed:\n{res.stderr}"
    assert csv_file.exists()
