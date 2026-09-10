# DubGraft

[![CI](https://github.com/NightCorpse/DubGraft/actions/workflows/ci.yml/badge.svg)](https://github.com/NightCorpse/DubGraft/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dubgraft)](https://pypi.org/project/dubgraft/)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
[![License: GPL v3 only](https://img.shields.io/badge/License-GPLv3--only-blue.svg)](LICENSE)

DubGraft aligns a selected audio track from one media release with the timeline
of another release, then adds the synchronized track while preserving the
Target's existing streams.

It is designed for cases where a desired dub is available in one release while
another offers better video quality. Shared music, effects, foley, and ambience
across different mixes are used to estimate their temporal relationship even
when the dialogue differs.

> [!NOTE]
> The core alignment workflows are functional and tested. Container
> compatibility, high-channel-count audio, and immersive formats remain areas
> of active development.

## Contents

- [What DubGraft does](#what-dubgraft-does)
- [Use cases](#use-cases)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Source, Target, and Output](#source-target-and-output)
- [Examples](#examples)
- [CLI reference](#cli-reference)
- [Defaults and tuning](#defaults-and-tuning)
- [Output strategies](#output-strategies)
- [Containers and platforms](#containers-and-platforms)
- [Audio safety](#audio-safety)
- [Known limitations](#known-limitations)
- [Planned features](#planned-features)
- [Language-code data](#language-code-data)
- [Contributing](#contributing)
- [Development transparency](#development-transparency)
- [License and responsible use](#license-and-responsible-use)

## What DubGraft does

Different releases of the same film, episode, or other video may have different
encodes, resolutions, containers, frame rates, introductions, or starting
timestamps. A desired dub may exist only in one release while another release
has the preferred video.

DubGraft compares selected audio tracks from both releases, finds matching
events across their timelines, and determines whether they are:

| Result | Meaning | Audio handling |
|---|---|---|
| `direct` | The timelines already match within the configured tolerance. | Stream copy, with no offset applied. |
| `static` | The timelines run at the same speed with a constant offset. | Trim or delay, then stream copy. |
| `drift` | The timing difference changes approximately linearly. | Continuous retiming and E-AC-3 encoding. |
| `inconclusive` | The evidence is not strong or consistent enough. | No output is created. |

The Target supplies the final timeline and all streams that should remain in
the output. DubGraft appends only the selected Source audio track.

## Use cases

Examples include, but are not limited to:

- Adding a dubbed track from a 1080p release to a matching 4K release.
- Transferring audio between WEB-DL, Blu-ray, remux, or archival releases that
  share the same edit.
- Correcting a constant delay between two releases without re-encoding the
  Source audio.
- Correcting gradual linear timing drift, such as that caused by differences
  between 23.976 fps and 24 fps releases or by clock variations.
- Recovering a language track from an older recording for use with a better
  visual source.

DubGraft is not a media downloader and does not provide audio or video files.

## How it works

### Audio matching

DubGraft first uses FFprobe to inspect both files and identify the selected
audio streams. For analysis only, it decodes short windows to mono PCM at
22,050 Hz. This temporary analysis representation does not determine the
quality or channel layout of the final track.

At regular points on the Target timeline, DubGraft:

1. Extracts a short Target audio fingerprint.
2. Extracts a larger Source window around the expected time.
3. Uses cross-correlation to find the strongest position of the fingerprint in
   that Source window.
4. Measures how prominent the strongest correlation peak is relative to the
   average absolute correlation.
5. Keeps candidates that meet the confidence threshold.

The implementation uses
[`scipy.signal.correlate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)
for cross-correlation.

### Confidence

For correlation values `C`, confidence is calculated as:

```text
confidence = abs(strongest peak) / mean(abs(C))
```

Confidence measures how prominently the strongest absolute correlation peak
stands out from the average absolute correlation in that search. A value of
`100` means the selected peak is 100 times larger than that average. DubGraft
uses this value to filter and rank candidate matches.

DubGraft also evaluates anchor count, temporal coverage, offset consistency,
regression residuals, and Source duration when classifying the result.

### Distributed anchors

Candidates are ordered by confidence. DubGraft selects the strongest candidates
while enforcing a minimum distance between them on the Target timeline. This
reduces local clustering and provides evidence from different parts of the
media.

The automatic requested anchor count is:

```text
anchors = max(4, round(target duration / 240 seconds))
```

The automatic minimum gap is:

```text
anchor gap = max(20, int(0.6 * target duration / anchors)) seconds
```

These are practical project defaults, not universal or theoretically optimal
values.

### Timeline model

Each selected anchor relates a time in the Target to a matching time in the
Source. DubGraft fits the linear model:

```text
source_time = slope * target_time + intercept
```

The implementation uses
[`numpy.polyfit`](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)
for the linear fit. `intercept` describes an initial offset, while `slope`
describes a progressive timing difference. A single line cannot represent
arbitrary inserted, removed, or reordered scenes.

## Requirements

- Python 3.11 or newer.
- FFmpeg and FFprobe available through `PATH`.
- Windows or Linux.
- Enough free disk space beside the Output for temporary rendering files.

Install FFmpeg through your operating system or from the
[official FFmpeg download page](https://ffmpeg.org/download.html). For example:

```bash
# Debian or Ubuntu
sudo apt install ffmpeg

# Windows with Chocolatey
choco install ffmpeg
```

Verify that both tools are visible:

```bash
ffmpeg -version
ffprobe -version
```

## Installation

Install DubGraft from PyPI:

```bash
python -m pip install dubgraft
```

Verify the installed command:

```bash
dubgraft --version
```

## Quick start

Inspect both releases to list their streams and indices:

```bash
dubgraft inspect "dubbed-release.mkv" "reference-release.mkv"
```

If each input contains only one audio track, DubGraft selects it automatically:

```bash
dubgraft "dubbed-release.mkv" "reference-release.mkv" "output.mkv"
```

If either file contains multiple audio tracks, use the indices shown by
`inspect` to identify the Source and Target audio:

```bash
dubgraft "dubbed-release.mkv" "reference-release.mkv" "output.mkv" --source-audio 2 --target-audio 1
```

Start with the defaults. Use `--analyze-only` and inspect the diagnostics before
tuning matching parameters.

After processing, play the Output and verify synchronization at several points.
Automated analysis and structural validation do not replace perceptual review.

## Source, Target, and Output

The three paths have fixed roles:

| Role | Purpose |
|---|---|
| Source | Provides the audio searched during matching; the selected audio and its track metadata are carried into the result. |
| Target | Provides the fingerprints and defines the final timeline; all its streams and metadata are retained. |
| Output | Combines the Target with the synchronized Source audio as a new track. |

Additional Source streams or metadata can be handled separately with a
general-purpose muxing tool.

The positional order is always:

```text
dubgraft SOURCE TARGET OUTPUT
```

Equivalent explicit and mixed forms are accepted:

```bash
dubgraft --source source.mkv --target target.mkv --output output.mkv
dubgraft source.mkv target.mkv --output output.mkv
```

If Output has no extension, it inherits the Target extension:

```bash
dubgraft source.mp4 target.mkv output
# Creates output.mkv
```

An explicit Output extension requests that container. DubGraft aborts if FFmpeg
cannot safely preserve the Target streams or add the selected Source audio to
it.

Rendering occurs in a temporary file. DubGraft probes and validates the result
before publishing it as Output.

## Examples

### Add a dub and preserve its metadata

By default, the selected Source track's title, language, and dispositions are
carried to the added track:

```bash
dubgraft source.mkv target.mkv output.mkv -S 2 -T 1
```

### Override the added track metadata

```bash
dubgraft source.mkv target.mkv output.mkv -S 2 -T 1 --language pt --track-name "Brazilian Portuguese"
```

`pt` is accepted and stored as its ISO 639-2/B form, `por`.

### Analyze without creating media

```bash
dubgraft source.mkv target.mkv --analyze-only --print-report
```

### Save machine-readable diagnostics

```bash
dubgraft source.mkv target.mkv --analyze-only --report analysis.json
```

### Process and keep a diagnostic log

```bash
dubgraft source.mkv target.mkv output.mkv --report result.json --log dubgraft.log
```

### Search for a large constant offset

The default search radius is 75 seconds on each side of the expected time. If
the releases differ by more than that near every scan point, increase it:

```bash
dubgraft source.mkv target.mkv output.mkv --search-radius 120
```

### Examine more points on a difficult release

Reducing the scan step examines the Target more densely, at the cost of a
longer analysis:

```bash
dubgraft source.mkv target.mkv output.mkv --scan-step 10
```

### Replace an existing Output

```bash
dubgraft source.mkv target.mkv output.mkv --overwrite
```

## CLI reference

### Commands and paths

| Argument | Default | Description |
|---|---|---|
| `dubgraft SOURCE TARGET OUTPUT` | - | Process using positional paths. |
| `dubgraft inspect MEDIA [...]` | - | Inspect one or more files without modifying them. |
| `-s`, `--source PATH` | - | Explicit Source path. |
| `-t`, `--target PATH` | - | Explicit Target path. |
| `-o`, `--output PATH` | - | Explicit Output path. Without an extension, uses the Target extension. |
| `-S`, `--source-audio INDEX` | Automatic with one track | Global index of the Source audio to add. |
| `-T`, `--target-audio INDEX` | Automatic with one track | Global index of the Target audio used for matching. |
| `-y`, `--overwrite` | Off | Allow replacement of an existing Output or Report. |
| `-h`, `--help` | - | Show command help. |
| `--version` | - | Show the installed DubGraft version. |

### Added-track metadata

| Argument | Default | Description |
|---|---|---|
| `--track-name NAME` | Source title | Override the added track title. |
| `--language CODE` | Source language | Accept ISO 639-1 or ISO 639-2 and store ISO 639-2/B. |

### Matching and timeline analysis

| Argument | Default | Description |
|---|---:|---|
| `-F`, `--fingerprint-size SECONDS` | `6` | Duration of each Target sample used for correlation. |
| `-p`, `--scan-step SECONDS` | `20` | Interval between examined Target timeline points. |
| `-r`, `--search-radius SECONDS` | `75` | Source search distance on each side of the expected time. |
| `-c`, `--min-confidence VALUE` | `60` | Minimum correlation-peak prominence accepted as a candidate. |
| `-a`, `--anchors NUMBER` | Automatic | Target number of candidates selected as distributed anchors; the actual count may be lower. Minimum: `4`. |
| `-g`, `--anchor-gap SECONDS` | Automatic | Minimum spacing between selected anchors, preventing clustering and encouraging broader timeline coverage. |
| `-d`, `--direct-limit MS` | `20` | Largest median offset left uncorrected as `direct`. |
| `-f`, `--force` | Off | Allow only advanced analysis values outside recommended limits. |

`--min-confidence` below `20` and `--direct-limit` above `50` require
`--force`.

### Diagnostics and terminal output

| Argument | Default | Description |
|---|---|---|
| `--analyze-only` | Off | Analyze synchronization without creating Output. |
| `--report PATH` | None | Write a JSON report to the specified path. |
| `--print-report` | Off | Print candidates, anchors, formulas, and model diagnostics. |
| `-v`, `--verbose` | Off | Show paths, streams, durations, decisions, and other details. |
| `-q`, `--quiet` | Off | Suppress progress and the normal final summary, but not errors or an explicitly printed report. |
| `--log PATH` | None | Append timestamped diagnostics and external commands to a log file. |

`--verbose` and `--quiet` are mutually exclusive. Output, Report, and Log parent
directories must already exist.

## Defaults and tuning

Defaults are intended as a safe starting point. Tuning cannot make a single
linear model represent releases with arbitrary internal edits.

| Situation | Possible adjustment | Trade-off |
|---|---|---|
| The constant offset may exceed 75 seconds. | Increase `--search-radius`. | Larger windows require more decoding and correlation work. |
| Too few timeline positions contain useful shared audio. | Reduce `--scan-step`. | More FFmpeg operations and a longer analysis. |
| Shared events need more context. | Increase `--fingerprint-size` moderately. | Longer shared segments are required and correlation costs more. |
| Valid-looking peaks fall below confidence 60. | Lower `--min-confidence` cautiously. | Weak peaks increase the risk of false matches. |
| Strong candidates are rejected for being too close. | Reduce `--anchor-gap`. | Anchors may cluster and provide worse timeline coverage. |
| More distributed evidence is desired. | Increase `--anchors`. | More anchors can be selected only when enough valid candidates meet the spacing requirement. |
| Offsets up to 20 ms are classified as direct by default. | Increase `--direct-limit` to classify larger offsets as direct. | Larger uncorrected offsets may become perceptible. |
| Cuts or reordered scenes occur within the media. | No tuning can represent the resulting timeline discontinuity. | Differences limited to the beginning or end may still align if the remaining timeline is consistent. |

### Internal classification thresholds

These defaults affect classification but are not currently exposed as CLI
options:

| Check | Default |
|---|---:|
| Minimum selected anchors | `4` |
| Normal Target timeline coverage | `60%` |
| Strict fallback coverage | `50%` |
| Stable-offset tolerance | `50 ms` |
| Required stable-offset ratio for non-drift results | `80%` |
| Total drift required for `drift` | More than `50 ms` over the Target duration |
| Maximum RMS regression residual | `50 ms` |

Coverage between 50% and 60% is accepted only by a strict fallback that also
requires RMS residual at most 10 ms, maximum residual at most 20 ms, and a
predicted Source duration within 100 ms of the measured duration.

## Output strategies

### Direct

Offsets within `--direct-limit` are considered small enough to leave
uncorrected. The selected Source audio is appended with stream copy.

### Static offset

When the timelines have the same speed but a constant offset, DubGraft trims or
delays the selected Source audio and keeps stream copy. The precision of a
stream-copy trim can depend on codec and container timestamp boundaries.

### Linear drift

When the fitted timing difference changes progressively, DubGraft applies one
continuous FFmpeg retiming operation based on the fitted slope and intercept.
The reconstructed track is encoded once as E-AC-3 at 640 kb/s, then copied into
the final container. It is padded or trimmed to the Target duration.

The retiming path uses FFmpeg's
[`atempo`](https://ffmpeg.org/ffmpeg-filters.html#atempo) filter.

## Containers and platforms

### Officially supported containers

| Container | Status | Notes |
|---|---|---|
| Matroska (`.mkv`) | Officially supported | Recommended when broad stream and codec compatibility is needed. |
| MPEG-4 (`.mp4`) | Officially supported | Codec and metadata compatibility still depend on FFmpeg and the streams involved. |
| M4V (`.m4v`) | Experimental | Receives MP4-family handling but lacks dedicated integration coverage. |
| QuickTime (`.mov`) | Experimental | Receives MP4-family handling but lacks dedicated integration coverage. |
| AVI, WebM, and others | Experimental | Attempted through FFmpeg preflight; preservation is not guaranteed. |

MKV and MP4 are tested as inputs and outputs, including cross-container use.
Container support never implies that every audio, subtitle, attachment, or data
codec is valid in that container. DubGraft runs preflight checks before
analysis and before rendering the chosen audio strategy.

The current drift result is E-AC-3 and therefore cannot be written to WebM.

### Platforms

| Platform | Status |
|---|---|
| Linux | Tested in CI |
| Windows | Tested in CI |
| macOS | Not currently tested or officially supported |

## Audio safety

DubGraft distinguishes stream-copy synchronization from drift correction:

| Source audio | Direct/static | Drift |
|---|---|---|
| Conventional lossy audio up to 5.1 | Stream copy when the container accepts it | E-AC-3 640 kb/s, preserving known channels, layout, and sample rate |
| Lossless audio | Stream copy when the container accepts it | Rejected to avoid an implicit lossy conversion |
| 7.1 audio | Stream copy when the container accepts it | Rejected; the current drift backend supports at most six channels |
| Detected Atmos or other object audio | Bitstream can remain intact through stream copy | Rejected because filtering would discard object metadata |
| Unknown or unclassified audio | Stream copy may be attempted | Rejected rather than converted implicitly |

After drift reconstruction, DubGraft probes the result and verifies that the
channel count, channel layout, and sample rate still match the selected Source
track. It does not silently downmix.

Atmos and 7.1 are separate properties. Preserving eight channel beds would not
by itself preserve Atmos objects and metadata. Re-authoring Atmos after timing
changes requires an appropriate master and specialized licensed tooling; it
cannot be safely reconstructed by the current FFmpeg path.

## Known limitations

- Matching depends on enough shared music, effects, foley, or ambience between
  the selected mixes. Dialogue is not compared semantically.
- A single linear model supports direct synchronization, a constant offset, or
  approximately linear drift. It does not support arbitrary internal cuts,
  inserted scenes, removed scenes, changing offsets, or reordered sequences.
- Different masters or heavily altered mixes may produce too few reliable
  matches.
- Confidence is peak prominence, not a calibrated probability of correctness.
- Anchor selection prevents nearby clustering but does not guarantee uniform
  distribution or a globally optimal set.
- The regression is not confidence-weighted. A stable minority outlier may be
  excluded, but the fit is not a general robust or piecewise regression.
- Static stream-copy trimming is limited by codec/container timestamp behavior.
- Drift always re-encodes the selected Source audio as E-AC-3 640 kb/s.
- Drift currently supports at most 5.1 and rejects lossless, detected immersive,
  and unclassified audio.
- Only MKV and MP4 are officially supported in the initial release.
- FFmpeg compatibility varies by build, version, codec, and container.
- Post-mux validation checks stream structure, metadata, chapters, timing, audio
  layout, and relevant video properties, but it does not fully decode the media
  or judge perceived synchronization.

## Planned features

Planned work includes:

- Start-offset discovery for incompatible introductions.
- Reverse-direction anchor scanning.
- Exposing the analysis sample rate after broader validation.
- A safe FLAC 7.1 drift path for Matroska.
- Independent consent flags for downmix, lossless-to-lossy conversion, and loss
  of immersive metadata.
- Dedicated integration coverage and possible official support for additional
  containers.
- Optional integration with appropriate licensed tools for Atmos re-authoring
  when a suitable master is available.

These items are directions, not commitments to a specific release.

## Language-code data

`--language` accepts ISO 639-1, ISO 639-2/B, and ISO 639-2/T codes. Inputs are
case-insensitive and normalized to ISO 639-2/B for the added track:

| Input | Stored value |
|---|---|
| `pt` or `por` | `por` |
| `fr`, `fra`, or `fre` | `fre` |
| `de`, `deu`, or `ger` | `ger` |

DubGraft bundles a transformed snapshot of the Library of Congress
[ISO 639-2 code list](https://www.loc.gov/standards/iso639-2/ISO-639-2_utf-8.txt),
where the Library of Congress serves as the ISO 639-2 Registration Authority.
The original UTF-8, pipe-delimited data was converted to CSV with named columns;
its 487 records and ordering were retained. The bundled snapshot is available
at [`src/dubgraft/data/iso-639-2.csv`](src/dubgraft/data/iso-639-2.csv).

This attribution records the source and transformation of the data. It does not
imply endorsement by the Library of Congress or ISO.

## Contributing

Contributions, bug reports, and suggestions are welcome.

Create a development environment:

```bash
# Linux
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

```powershell
# Windows
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Ensure FFmpeg and FFprobe are available, then run the same core checks used by
CI:

```bash
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src/dubgraft
.venv/bin/python -m pytest
.venv/bin/python -m build
```

Media tests should use small synthetic fixtures generated during the test. Do
not commit copyrighted media, personal releases, logs containing private paths,
or generated output files.

Contributions that change matching behavior should include evidence for direct,
static, drift, and inconclusive outcomes where applicable. Contributions that
change muxing should verify stream preservation and failure without partial
publication.

## Development transparency

AI-assisted development tools were used during portions of implementation,
documentation, and review. All changes were reviewed and tested before
inclusion.

## License and responsible use

DubGraft is licensed under the
[GNU General Public License v3.0 only](LICENSE).

Use DubGraft only with media you are legally permitted to access and modify. You
are responsible for complying with applicable copyright and licensing
requirements.
