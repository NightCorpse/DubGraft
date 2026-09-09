import shutil
import subprocess
from pathlib import Path

import pytest

from dubgraft.config import ProcessingConfig
from dubgraft.matching import TimelineAnalysis, TimelineKind
from dubgraft.media import probe_media, select_audio_stream
from dubgraft.processing import (
    mux_drift_audio,
    mux_source_audio,
    validate_mux_compatibility,
    validate_render_compatibility,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are required",
)


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *arguments],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def synthetic_media(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=48000:duration=5",
        "-c:a",
        "eac3",
        "-b:a",
        "192k",
        "-ac",
        "6",
        "-metadata:s:a:0",
        "language=por",
        "-metadata:s:a:0",
        "title=Dublado",
        str(source),
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=size=64x64:rate=24:duration=4",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=4",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "ffv1",
        "-c:a",
        "pcm_s16le",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=English",
        str(target),
    )
    return source, target


@pytest.mark.parametrize(
    "kind", [TimelineKind.DIRECT, TimelineKind.STATIC, TimelineKind.DRIFT]
)
def test_synthetic_direct_static_and_drift_outputs(
    kind: TimelineKind,
    synthetic_media: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    source, target = synthetic_media
    source_info = probe_media(source)
    target_info = probe_media(target)
    source_audio = select_audio_stream(source_info)
    output = tmp_path / f"{kind.value}.mkv"
    config = ProcessingConfig(
        source,
        target,
        output,
        language="eng",
        track_name="Dubbed",
    )
    progress: list[tuple[str, float, float]] = []

    if kind is TimelineKind.DRIFT:
        analysis = TimelineAnalysis(
            kind,
            0.1,
            0.999,
            0.1,
            0,
            0,
            0.8,
            0,
            -0.004,
        )
        validated = mux_drift_audio(
            config,
            target_info,
            source_audio,
            analysis,
            progress=lambda *values: progress.append(values),
        )
    else:
        validated = mux_source_audio(
            config,
            target_info,
            source_audio,
            offset=0 if kind is TimelineKind.DIRECT else 0.25,
            source_duration=source_info.duration,
            progress=lambda *values: progress.append(values),
        )

    expected_phases = (
        {"audio reconstruction", "mux"}
        if kind is TimelineKind.DRIFT
        else {"audio trim", "mux"}
        if kind is TimelineKind.STATIC
        else {"mux"}
    )
    assert {phase for phase, _, _ in progress} == expected_phases
    for phase in expected_phases:
        values = [
            (completed, total) for name, completed, total in progress if name == phase
        ]
        assert values[-1][0] == values[-1][1]
        assert values == sorted(values)
    if kind is TimelineKind.STATIC:
        assert progress[0][2] == source_info.duration

    assert validated.path == output.resolve()
    output_info = probe_media(output)
    assert output_info.duration == pytest.approx(target_info.duration, abs=0.05)
    assert [stream.kind for stream in output_info.streams] == [
        "video",
        "audio",
        "audio",
    ]
    assert output_info.streams[0].codec == "ffv1"
    assert output_info.streams[1].codec == "pcm_s16le"
    assert output_info.streams[2].codec == "eac3"
    assert output_info.streams[2].channels == source_audio.channels == 6
    assert output_info.streams[2].language == "eng"
    assert output_info.streams[2].title == "Dubbed"
    if kind is TimelineKind.DRIFT:
        assert output_info.streams[2].bit_rate == pytest.approx(640_000, rel=0.02)

    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("source_extension", "target_extension", "output_extension"),
    [
        ("mkv", "mkv", "mkv"),
        ("mp4", "mp4", "mp4"),
        ("mkv", "mp4", "mp4"),
        ("mp4", "mkv", "mkv"),
    ],
)
def test_container_matrix_preserves_track_metadata(
    source_extension: str,
    target_extension: str,
    output_extension: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / f"source.{source_extension}"
    target = tmp_path / f"target.{target_extension}"
    output = tmp_path / f"output.{output_extension}"
    metadata = tmp_path / "chapters.ffmeta"
    metadata.write_text(
        ";FFMETADATA1\ntitle=Target\n[CHAPTER]\nTIMEBASE=1/1000\n"
        "START=0\nEND=500\ntitle=Opening\n",
        encoding="utf-8",
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=48000:duration=1",
        "-c:a",
        "aac",
        "-metadata:s:a:0",
        "language=por",
        "-metadata:s:a:0",
        f"{'handler_name' if source_extension == 'mp4' else 'title'}=Portuguese",
        str(source),
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=size=16x16:rate=24:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1",
        "-f",
        "ffmetadata",
        "-i",
        str(metadata),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map_metadata",
        "2",
        "-map_chapters",
        "2",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        f"{'handler_name' if target_extension == 'mp4' else 'title'}=English",
        "-disposition:a:0",
        "default",
        str(target),
    )
    source_info = probe_media(source)
    target_info = probe_media(target)
    source_audio = select_audio_stream(source_info)
    config = ProcessingConfig(source, target, output)

    validate_mux_compatibility(config, target_info)
    validate_render_compatibility(
        config, target_info, source_audio, TimelineKind.DIRECT
    )
    validated = mux_source_audio(
        config,
        target_info,
        source_audio,
        offset=0,
        source_duration=source_info.duration,
    )

    audios = [stream for stream in validated.streams if stream.kind == "audio"]
    assert validated.title == "Target"
    assert len(validated.chapters) == 1
    assert validated.chapters[0].title == "Opening"
    assert audios[0].language == "eng"
    assert audios[0].title == "English"
    assert audios[0].dispositions == ("default",)
    assert audios[1].language == "por"
    assert audios[1].title == "Portuguese"


def test_mp4_cover_art_survives_mux(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mp4"
    output = tmp_path / "output.mp4"
    cover = tmp_path / "cover.jpg"
    metadata = tmp_path / "chapters.ffmeta"
    metadata.write_text(
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=500\ntitle=Opening\n",
        encoding="utf-8",
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:size=24x36",
        "-frames:v",
        "1",
        str(cover),
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=48000:duration=1",
        "-c:a",
        "aac",
        str(source),
    )
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=size=16x16:rate=24:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1",
        "-i",
        str(cover),
        "-f",
        "ffmetadata",
        "-i",
        str(metadata),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:v",
        "-map_metadata",
        "3",
        "-map_chapters",
        "3",
        "-c:v:0",
        "mpeg4",
        "-c:a",
        "aac",
        "-c:v:1",
        "mjpeg",
        "-disposition:v:1",
        "attached_pic",
        str(target),
    )
    source_info = probe_media(source)
    target_info = probe_media(target)
    source_audio = select_audio_stream(source_info)
    config = ProcessingConfig(source, target, output)

    validated = mux_source_audio(
        config,
        target_info,
        source_audio,
        offset=0,
        source_duration=source_info.duration,
    )

    covers = [stream for stream in validated.streams if stream.attached_picture]
    assert len(covers) == 1
    assert covers[0].codec == "mjpeg"
    assert "attached_pic" in covers[0].dispositions
