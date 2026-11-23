# SongKey

**Find the tempo (BPM) and musical key of your songs, see them as a waveform, and mark where to mix in and out.**
A free, open-source desktop app for DJs, written in Python. Think of it as a free alternative to Mixed In Key.


<img width="1024" height="741" alt="image" src="https://github.com/user-attachments/assets/59e02429-edb1-4f23-bbb5-50a63ded9eb2" />


<img width="1974" height="703" alt="image" src="https://github.com/user-attachments/assets/d7ea57f0-3906-4bf4-a2c9-24a4168693d3" />


---

## What it does

- **BPM**: measures the tempo of any song, with decimals when the song really has them (e.g. 133.9).
- **Key**: finds the musical key and shows it in **Camelot** notation (8A, 9B, ...), with the
  compatible keys for harmonic mixing listed underneath.
- **Plain-language results**: *"128 BPM, dance tempo (sure)"*, *"A minor (darker mood) - Camelot 8A"*.
  When two keys are close it says so honestly: *"not sure - could also be C minor (5A)"*.
- **Any audio format**: MP3, WAV, FLAC, AIFF, M4A/AAC, OGG, Opus, WMA... anything FFmpeg can read.
  Drag & drop files or whole folders onto the window.
- **Waveform**: green, Audacity-style, stereo, with zoom from the whole track down to single samples,
  a beat grid, and a whole-track overview strip.
- **Spectrum**: spectrum over time, average sound balance, and a "notes used" chart.
- **Mix points**: automatically places **IN**, **DROP**, **BREAK**, **OUT** and **END** marks on the bar grid.
  Drag them, nudge them by 10 ms or by one beat, or add your own. Adjusted marks are kept even if you
  re-analyse the song.
- **Big mark editor**: a separate, almost full-screen window with the waveform and a sharp, zoomable
  spectrogram lane for fine-tuning marks.
- **Built-in player**: click anywhere on the waveform to play from there; jump from mark to mark.
- **Tap BPM**: tap the space bar along with the song to check or set the tempo by ear.
  Detects missed beats and double taps, and tells you if the app made a typical mistake
  (half/double tempo, 4:3, 2:3).
- **Library**: songs you choose to keep go into a SQLite database together with the file's
  details (artist, title, mix/remix, remixer, genre, album, year, label, format, bitrate, size).
  Everything can be edited, also for several songs at once. Search, sort, export to CSV, and filter
  to **songs that mix well with the selected one** (compatible key and tempo).
- **Two analysis engines**: the built-in one (always available) and, optionally,
  [Essentia](https://essentia.upf.edu/) as a second opinion.
- **Runs on old hardware**: no SSE4/AVX needed when installed from your distribution's packages.

---

## Installation

You need **Python 3.9 or newer**, **Tk** for Python and **FFmpeg** (which also provides
`ffplay`, used for playback).

### Option A - Debian, Ubuntu, Devuan, Linux Mint (recommended)

Everything comes from your distribution, nothing from pip:

```bash
sudo apt install python3-tk python3-numpy python3-scipy python3-matplotlib \
                 python3-pil python3-pil.imagetk ffmpeg tkdnd
```

This is also **the way to go on older CPUs** (AMD Athlon II / Phenom and similar, without SSE4.2):
the distribution's packages run on any x86-64 CPU.

### Option B - pip

Install Tk and FFmpeg with your system's package manager, then:

```bash
pip3 install -r requirements.txt
```

Notes:

- On **Debian 12+, Ubuntu 23.04+ and Devuan 5+**, pip refuses to install into the system Python
  ("externally-managed-environment"). Use Option A there - forcing pip with
  `--break-system-packages` can replace your system's own numpy/scipy and break other programs.
- On **CPUs without SSE4.2**, the newest numpy from pip crashes with *"Illegal instruction"*.
  Use Option A.
- Drag & drop needs the `tkdnd` Tcl package (Option A installs it). On other systems you can
  `pip3 install tkinterdnd2` instead. Without either, use the **Open files / Open folder** buttons.

### Optional - Essentia engine

SongKey runs fine without it. If you want Essentia as a second engine:

- **Modern CPU, pip-based setup:** `pip3 install essentia`
- **Debian/Devuan with system packages (Option A), or an older CPU:** install a version that does
  not pull in a new numpy:

  ```bash
  sudo apt install python3-six python3-yaml
  sudo pip install --no-deps --break-system-packages essentia==2.1b6.dev1110
  ```

  `--no-deps` is essential: without it, pip replaces your system's numpy (and on older CPUs the new
  numpy crashes with "Illegal instruction"). Newer Essentia versions require numpy 1.25 or newer.

Then pick **Essentia** under **Engine** (bottom-right of the window). SongKey first runs a quick
test analysis in a separate process; if Essentia doesn't work on your computer, you get a message
and the built-in engine stays active - the app never crashes because of it.

---

## Usage

```bash
python3 songkey.py
```

1. **Drop audio files or folders** onto the window (or use **Open files** / **Open folder**).
2. Each song shows its waveform right away, then BPM, key and mix points a few seconds later.
   New songs appear under **Just analysed**.
3. Press **Add to library** to keep a song: SongKey reads the file's details into a form you can
   correct before saving. Tick *"Ask me to add each song after analysis"* to be asked automatically.
4. Adjust the mix marks if needed (drag, nudge, or open the **Big editor**). Changes are saved
   to the library straight away.

### Command line

```bash
python3 songkey.py --cli ~/Music/new/             # analyse a folder, print results
python3 songkey.py --cli song.mp3 --add           # ...and add them to the library
python3 songkey.py --db /path/to/songs.db         # use another library file
python3 songkey.py --engine essentia              # start with the Essentia engine
python3 songkey.py --bpm-range 88-180             # fold tempos into this range (e.g. for drum & bass)
```

### Keyboard and mouse

| Where | Action | What it does |
|---|---|---|
| Anywhere | `Space` | Play / pause |
| Anywhere | `I` / `O` | Put an IN / OUT mark at the playing position |
| Waveform | Click | Play from there |
| Waveform | Drag a mark | Move it (snaps to the beat if *Snap to beat* is on) |
| Waveform | Right-click | Add / move / remove marks, zoom in, reset marks |
| Waveform | Mouse wheel | Zoom in / out around the pointer |
| Waveform | `Shift` + wheel | Scroll left / right |
| Waveform | `Left` / `Right` | Move the selected mark 10 ms |
| Waveform | `Shift` + `Left` / `Right` | Move the selected mark one beat |
| Waveform | `Delete` | Delete the selected mark |
| Waveform | `+` / `-` | Zoom in / out |
| Tap BPM window | `Space` | Tap the beat |
| Tap BPM window | `Enter` | Play / pause |
| Song list | Double-click / `Enter` | Edit the song's details |

Hover over any button for a short explanation.

### The mix marks

| Mark | Meaning |
|---|---|
| **IN** | First beat - start mixing this track in here |
| **DROP** | Full energy begins - the incoming track should be fully in |
| **BREAK** | Breakdown - an early place to mix out |
| **OUT** | Outro begins - start bringing in the next track |
| **END** | Last beat |

Marks you moved or added yourself show a `*`. **Reset marks** brings back the automatic ones.

---

## How it works (short version)

- **Decoding**: FFmpeg turns any format into raw audio.
- **BPM**: an onset ("hit") envelope is built from the sound, and its autocorrelation shows how
  the music repeats. Candidate tempos are checked at 1, 2, 3 and 4 beats on a fine, sub-frame grid,
  which rejects the classic 2:3 and 3:4 mistakes caused by dotted rhythms and syncopated basslines.
  Results within 0.10 of a whole number are rounded (most produced music is at an exact tempo).
- **Key**: a 12-note profile (chroma) is built with tuning correction and drum suppression, then
  compared with major and minor key profiles. A second step checks the *home note* using the bass
  line, the fifth and the leading tone - this catches Dorian and Mixolydian songs (very common in
  dance music) that plain profile matching gets wrong.
- **Mix points**: beats -> bars -> energy in bass / mids / highs per bar -> section changes on
  4-bar lines -> IN / DROP / BREAK / OUT / END.
- **Library**: a single SQLite file (`songs.db` next to `songkey.py` by default) holding the
  results, the file details, your marks, and a compact copy of each song's waveform and spectrum,
  so songs open instantly even if the audio file has moved.

### Accuracy

On the author's test set (synthetic songs built to cover common rhythms and keys, including
Dorian and Mixolydian tracks), SongKey gets **51 of 52 tempos** and **81 of 84 keys** right.
Real music is messier: half/double tempo (e.g. drum & bass at 87 vs 174) and keys that sit between
two candidates are the typical cases. That's what the honest *"not sure / could also be"* hints,
the **Tap BPM** tool and the optional Essentia engine are for. Bug reports with the song name,
what SongKey said and what is correct are very welcome.

---

## Files

| File | What it is |
|---|---|
| `songkey.py` | The whole application (one file) |
| `requirements.txt` | Python packages for the pip installation |
| `songs.db` | Your library (created on first use; change with `--db`) |

---

## Roadmap

- Windows build
- Writing BPM / key tags and cue points back to the files, and a rekordbox XML export
- "What to play next" suggestions and playlist export
- Transition preview (play the end of one track mixed into the start of the next)
- Energy level (1-10)
- Analysing several songs at once on multi-core CPUs

---

## Credits

- **Idea, design and testing:** Pedro ([peterpt](https://github.com/peterpt))
- **Code:** written by **Claude Opus 5.5**, an AI model from [Anthropic](https://www.anthropic.com)
- **Built with:** Python, Tk, NumPy, SciPy, Matplotlib, Pillow, FFmpeg, and optionally Essentia

*Pedro had the idea - Claude wrote the code.*

## License

See [LICENSE](LICENSE).
