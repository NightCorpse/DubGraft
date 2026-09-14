import json
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


def test_mux_places_added_audio_before_subtitles_and_strips_stats_and_duplicate_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    target = tmp_path / "target.mkv"
    output = tmp_path / "output.mkv"
    sub1 = tmp_path / "sub1.srt"
    sub2 = tmp_path / "sub2.srt"
    sub1.write_text("1\n00:00:00,000 --> 00:00:01,000\nSub 1\n", encoding="utf-8")
    sub2.write_text("1\n00:00:00,000 --> 00:00:01,000\nSub 2\n", encoding="utf-8")

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=48000:duration=1",
        "-c:a",
        "eac3",
        "-b:a",
        "192k",
        "-metadata:s:a:0",
        "language=por",
        "-metadata:s:a:0",
        "title=Dublado",
        "-metadata:s:a:0",
        "BPS=192000",
        "-metadata:s:a:0",
        "NUMBER_OF_FRAMES=31",
        "-metadata:s:a:0",
        "NUMBER_OF_BYTES=24000",
        "-metadata:s:a:0",
        "_STATISTICS_TAGS=BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES",
        "-metadata:s:a:0",
        "_STATISTICS_WRITING_APP=mkvmerge v83.0",
        "-metadata:s:a:0",
        "_STATISTICS_WRITING_DATE_UTC=2026-09-14 01:25:04",
        "-disposition:a:0",
        "default",
        str(source),
    )

    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=size=64x64:rate=24:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1",
        "-i",
        str(sub1),
        "-i",
        str(sub2),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:s",
        "-map",
        "3:s",
        "-c:v",
        "ffv1",
        "-c:a",
        "pcm_s16le",
        "-c:s",
        "subrip",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=Original",
        "-disposition:a:0",
        "default",
        "-metadata:s:s:0",
        "title=English Subs",
        "-metadata:s:s:0",
        "language=eng",
        "-metadata:s:s:1",
        "title=Portuguese Subs",
        "-metadata:s:s:1",
        "language=por",
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

    assert validated.path == output.resolve()
    output_info = probe_media(output)

    assert [stream.kind for stream in output_info.streams] == [
        "video",
        "audio",
        "audio",
        "subtitle",
        "subtitle",
    ]
    assert output_info.streams[1].title == "Original"
    assert output_info.streams[2].title == "Dublado"
    assert output_info.streams[3].title == "English Subs"
    assert output_info.streams[4].title == "Portuguese Subs"

    assert "default" in output_info.streams[1].dispositions
    assert "default" not in output_info.streams[2].dispositions
    default_audios = [
        s
        for s in output_info.streams
        if s.kind == "audio" and "default" in s.dispositions
    ]
    assert len(default_audios) == 1

    probe_raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:1",
            "-show_entries",
            "stream_tags",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = json.loads(probe_raw.stdout)["streams"][0].get("tags", {})
    for key in (
        "_STATISTICS_TAGS",
        "_STATISTICS_WRITING_APP",
        "_STATISTICS_WRITING_DATE_UTC",
        "BPS",
        "NUMBER_OF_FRAMES",
        "NUMBER_OF_BYTES",
    ):
        assert key not in tags

    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
