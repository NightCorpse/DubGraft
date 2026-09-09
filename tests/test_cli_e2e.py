import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="FFmpeg and FFprobe are required",
    ),
]


@dataclass(frozen=True)
class SyntheticMedia:
    target: Path
    direct: Path
    static: Path
    drift: Path
    inconclusive: Path


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *arguments],
        check=True,
        capture_output=True,
        timeout=60,
    )


def dubgraft_executable() -> Path:
    name = "dubgraft.exe" if os.name == "nt" else "dubgraft"
    executable = Path(sys.executable).with_name(name)
    if not executable.is_file():
        pytest.fail(f"installed DubGraft entry point was not found: {executable}")
    return executable


def run_dubgraft(
    source: Path,
    target: Path,
    output: Path,
    report: Path,
    *,
    workdir: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [
            str(dubgraft_executable()),
            str(source),
            str(target),
            str(output),
            "-F",
            "0.25",
            "-p",
            "2",
            "-r",
            "1",
            "-c",
            "10",
            "-a",
            "10",
            "-g",
            "0",
            "--force",
            "--report",
            str(report),
            "--quiet",
        ],
        cwd=workdir,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


def decode_output(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=True,
        capture_output=True,
        timeout=60,
    )


@pytest.fixture(scope="session")
def synthetic_cli_media(tmp_path_factory: pytest.TempPathFactory) -> SyntheticMedia:
    directory = tmp_path_factory.mktemp("cli-e2e-media")
    master = directory / "master.wav"
    target = directory / "target.mkv"
    direct = directory / "source-direct.mkv"
    static = directory / "source-static.mkv"
    drift = directory / "source-drift.mkv"
    inconclusive = directory / "source-inconclusive.mkv"

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=color=white:seed=1729:amplitude=0.15:sample_rate=48000:duration=24",
        "-af",
        "highpass=f=80,lowpass=f=500,pan=stereo|c0=c0|c1=c0",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        "-c:a",
        "pcm_s16le",
        str(master),
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=black:size=64x64:rate=10:duration=24",
        "-i",
        str(master),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        "-c:v",
        "ffv1",
        "-c:a",
        "pcm_s16le",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=Reference",
        "-shortest",
        str(target),
    )
    for path, audio_filter in ((direct, None), (static, "adelay=350:all=1")):
        arguments = ["-i", str(master)]
        if audio_filter is not None:
            arguments.extend(["-af", audio_filter])
        arguments.extend(
            [
                "-map_metadata",
                "-1",
                "-threads",
                "1",
                "-c:a",
                "pcm_s16le",
                "-metadata:s:a:0",
                "language=por",
                "-metadata:s:a:0",
                "title=SyntheticDub",
                str(path),
            ]
        )
        run_ffmpeg(*arguments)
    run_ffmpeg(
        "-i",
        str(master),
        "-af",
        "asetrate=47850,aresample=48000",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-metadata:s:a:0",
        "language=por",
        "-metadata:s:a:0",
        "title=SyntheticDub",
        str(drift),
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo:d=24",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        "-c:a",
        "pcm_s16le",
        "-metadata:s:a:0",
        "language=por",
        "-metadata:s:a:0",
        "title=Unrelated",
        str(inconclusive),
    )
    return SyntheticMedia(target, direct, static, drift, inconclusive)


@pytest.mark.parametrize(
    ("source_name", "classification", "expected_offset"),
    [("direct", "direct", 0.0), ("static", "static", 0.35)],
)
def test_stream_copy_flow(
    source_name: str,
    classification: str,
    expected_offset: float,
    synthetic_cli_media: SyntheticMedia,
    tmp_path: Path,
) -> None:
    source = getattr(synthetic_cli_media, source_name)
    output = tmp_path / f"{classification}.mkv"
    report_path = tmp_path / f"{classification}.json"

    result = run_dubgraft(
        source,
        synthetic_cli_media.target,
        output,
        report_path,
        workdir=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["timeline"]["classification"] == classification
    assert report["timeline"]["median_offset_seconds"] == pytest.approx(
        expected_offset, abs=0.02
    )
    assert report["matching"]["anchor_count"] >= 4
    assert report["timeline"]["coverage"] >= 0.6
    assert report["processing"]["strategy"] == "stream_copy"
    assert report["processing"]["target_streams_preserved"] == 2

    output_probe = probe(output)
    streams = output_probe["streams"]
    assert [stream["codec_type"] for stream in streams] == [
        "video",
        "audio",
        "audio",
    ]
    assert [stream["codec_name"] for stream in streams] == [
        "ffv1",
        "pcm_s16le",
        "pcm_s16le",
    ]
    assert streams[2]["tags"]["language"] == "por"
    assert streams[2]["tags"]["title"] == "SyntheticDub"
    assert float(output_probe["format"]["duration"]) == pytest.approx(24, abs=0.1)
    decode_output(output)
    assert not list(tmp_path.glob(".dubgraft-*"))


def test_drift_flow(
    synthetic_cli_media: SyntheticMedia,
    tmp_path: Path,
) -> None:
    output = tmp_path / "drift.mkv"
    report_path = tmp_path / "drift.json"

    result = run_dubgraft(
        synthetic_cli_media.drift,
        synthetic_cli_media.target,
        output,
        report_path,
        workdir=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["timeline"]["classification"] == "drift"
    assert report["timeline"]["slope"] == pytest.approx(48000 / 47850, abs=0.001)
    assert abs(report["timeline"]["drift_over_duration_seconds"]) > 0.05
    assert report["processing"]["strategy"] == "eac3_640k"
    assert report["processing"]["target_streams_preserved"] == 2

    output_probe = probe(output)
    streams = output_probe["streams"]
    assert [stream["codec_type"] for stream in streams] == [
        "video",
        "audio",
        "audio",
    ]
    added_audio = streams[2]
    assert added_audio["codec_name"] == "eac3"
    assert added_audio["channels"] == 2
    assert added_audio["sample_rate"] == "48000"
    assert int(added_audio["bit_rate"]) == pytest.approx(640_000, rel=0.02)
    assert added_audio["tags"]["language"] == "por"
    assert added_audio["tags"]["title"] == "SyntheticDub"
    assert float(output_probe["format"]["duration"]) == pytest.approx(24, abs=0.1)
    decode_output(output)
    assert not list(tmp_path.glob(".dubgraft-*"))


def test_inconclusive_does_not_publish(
    synthetic_cli_media: SyntheticMedia,
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist.mkv"
    report_path = tmp_path / "inconclusive.json"

    result = run_dubgraft(
        synthetic_cli_media.inconclusive,
        synthetic_cli_media.target,
        output,
        report_path,
        workdir=tmp_path,
    )

    assert result.returncode == 1
    assert not output.exists()
    assert "inconclusive analysis" in result.stderr
    assert "No output was created" in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "inconclusive"
    assert report["timeline"]["classification"] == "inconclusive"
    assert report["timeline"]["reason"] == "no anchors were found"
    assert report["processing"]["rendered"] is False
    assert report["processing"]["output_validated"] is False
    assert report["processing"]["added_audio"] is None
    assert not list(tmp_path.glob(".dubgraft-*"))
