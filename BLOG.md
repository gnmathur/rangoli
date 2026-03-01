# I Vibe Coded an Entire Desktop App Without Writing a Single Line of Code

**By Gaurav Mathur | March 2026**

---

I've been writing Python for over two decades. I can navigate threading bugs in my sleep, I have opinions about ORMs, and I've shipped production code in more languages than I care to count. So when I say I built a fully-featured desktop application without writing a single line of code myself, understand that it wasn't because I *couldn't*. It was because I wanted to see what happens when a seasoned developer stops typing code and starts directing an AI to do it instead.

The result is **Rangoli** — a podcast transcription app with a polished GUI, dual transcription engines, speaker diarization, AI-powered analysis, and about 2,400 lines of Python. Every one of those lines was written by Claude. My job was to think, describe, review, and steer.

This is the story of how that went.

## The Problem I Wanted to Solve

I listen to a lot of podcasts. Technical interviews, research discussions, long-form conversations where someone says something brilliant at the 47-minute mark and I can never find it again. I wanted searchable transcripts. Not cloud-based, not subscription-based — local, on my Mac, against any podcast RSS feed.

Tools existed, but they were all CLI-only, or cloud services with per-minute pricing, or research demos with no UI. I wanted something I could actually *use* day to day: subscribe to a podcast, pick an episode, click Transcribe, and get a timestamped transcript saved to a database. Later, I added AI summarization because once you have a transcript, sending it to GPT for a summary is obvious.

The kind of app I'd normally spend a weekend building. Instead, I spent that weekend doing something different.

## What "Vibe Coding" Actually Looks Like

The term "vibe coding" gets thrown around a lot. For most people it seems to mean "I asked ChatGPT to write a function and pasted it in." That's not what I did. What I did was closer to being a very opinionated tech lead sitting next to a junior developer who happens to type at the speed of light and never gets tired.

My workflow looked like this:

1. **Describe what I want** — not in pseudocode, but in product terms. "I want a sidebar with podcast artwork, a main area with a paginated episode table, and a progress bar at the bottom."
2. **Review what comes back** — read the code, understand the choices, spot issues.
3. **Steer corrections** — "The context menu should disable Transcribe if a transcription is already in progress." "That lambda will fail because Python clears exception variables after the except block."
4. **Iterate on polish** — "The cursor shouldn't change to a hand pointer on the podcast list. Native macOS apps use the arrow cursor."

I never opened an editor. I never wrote a function. But I was doing *engineering* the entire time — making architectural decisions, catching bugs, enforcing consistency, choosing tradeoffs.

## Starting From Scratch: CLI First

I started with the CLI because it's the simplest expression of the core problem: give me an RSS feed, download an episode, transcribe it with Whisper, save the output.

I described the flow I wanted and Claude produced `podcast_transcriber.py` — about 400 lines that handle feed parsing, audio download, transcription with OpenAI Whisper, optional speaker diarization via pyannote.audio, and formatted text output.

The interesting part was the diarization integration. Whisper produces timestamped segments. pyannote produces a speaker timeline. Merging them requires an overlap algorithm — for each Whisper segment, find the diarization turn with the maximum time overlap and assign that speaker. It's O(S*T) but for podcast episodes with a few hundred segments and similar turns, it runs in milliseconds. Claude got this right on the first try, which frankly would have taken me a few iterations to get the edge cases right.

[SCREENSHOT: CLI output showing a transcribed episode with speaker labels and timestamps]

## The GUI: Where It Got Interesting

The CLI was a warm-up. The real test was the GUI. I chose CustomTkinter because I wanted a modern dark-themed interface without the weight of Qt or Electron. This is where directing an AI starts to diverge from just "prompting" — you're making dozens of micro-decisions that compound.

**Layout decisions:**
- PanedWindow with a resizable sash between sidebar and main area
- Podcast list with artwork icons loaded asynchronously
- Episode table with dynamic page sizing based on window height
- 4-stage progress pipeline with per-segment ETAs

[SCREENSHOT: Main application window showing the sidebar with podcast artwork, episode list with status indicators, and progress bar during transcription]

**Threading model:**
This is where my experience mattered most. A GUI that blocks during a 20-minute transcription is useless. I needed background threads for everything — transcription, feed refresh, model preloading, artwork loading — with all UI updates marshaled back to the main thread via `after()`. Claude understood this pattern, but I had to be specific about cancellation checkpoints (per download chunk, per transcription segment, before and after diarization) and cooperative signaling via `threading.Event`.

[SCREENSHOT: Transcription in progress with the progress bar showing segment count, audio time processed, and estimated time remaining]

**The macOS details:**
This is where I was pickiest. I wanted the menu bar to say "Rangoli", not "Python". That required modifying `CFBundleName` in the main bundle's info dictionary via CoreFoundation's C API through `ctypes` — and it has to happen *before* tkinter imports. I wanted arrow cursors on list items instead of hand pointers. I wanted the About dialog to show the app icon, version, copyright, and license. None of these are hard individually, but together they're the difference between "Python script with a GUI" and "Mac app."

[SCREENSHOT: macOS menu bar showing "Rangoli" with the File, View, and AI menus expanded]

[SCREENSHOT: About dialog showing the Rangoli icon, version, and author information]

## Dual Engine Support

One of my early decisions was supporting both OpenAI Whisper and faster-whisper. faster-whisper uses CTranslate2 with int8 quantization and is roughly 4x faster on CPU. But the two engines have completely different APIs — Whisper returns a dict with all segments at once, faster-whisper returns a lazy generator with a `TranscriptionInfo` object.

I wanted uniform progress reporting across both engines. For faster-whisper, it's natural — you iterate the generator one segment at a time and compute `seg.end / info.duration`. For OpenAI Whisper, Claude came up with a clever approach: redirect `sys.stdout` to a custom writer class during `model.transcribe(verbose=True)`, parse the `[MM:SS.mmm --> MM:SS.mmm]` output via regex, and extract per-segment progress. Same UI, same ETA calculation, regardless of engine. The cancellation story even works — the stdout writer checks a `threading.Event` and raises an exception from within Whisper's `print()` call.

This is the kind of architectural decision I described at a high level — "I want both engines to show identical per-segment progress in the same format" — and Claude figured out the mechanism. I probably would have done the same thing, but it would have taken me longer to think through the stdout capture approach.

## Adding AI Analysis: The Cloud Layer

After a few days of using Rangoli for transcription, I wanted summaries. The transcripts were useful for searching, but for getting the gist of an episode, I wanted GPT to distill it down to key topics and takeaways.

This was the last major feature and it touched almost every layer:
- **Database**: New `analyses` table with the prompt and model used for reproducibility
- **GUI**: AI menu for prompt configuration, right-click "Analyze with AI" and "Show Analysis", a 3rd column panel for viewing results
- **Status progression**: Episodes now show (empty) -> Transcribed (green) -> Analyzed (purple) -> Analyzing... (orange, while in progress)

[SCREENSHOT: Episode list showing different status states — empty, "Transcribed" in green, "Analyzing..." in orange, and "Analyzed" in purple]

The design evolved through conversation. Initially the analysis panel opened immediately and showed "Analyzing..." — but that was disruptive when you just wanted to fire off an analysis and keep browsing. So I changed it: analysis runs silently in the background, a modal pops up when it's done (or on error), and you open the 3rd column on demand via "Show Analysis." Multiple episodes can be analyzed concurrently, each showing "Analyzing..." in orange in the status column.

[SCREENSHOT: Analysis complete dialog showing a formatted summary with key topics and takeaways]

[SCREENSHOT: The 3rd column analysis panel showing a formatted analysis with bold headers and bullet points]

One bug I caught during testing: the error handler used a lambda that captured the exception variable `e`, but Python clears exception variables when the `except` block exits to break reference cycles. The lambda would fire on the main thread later and crash with `NameError: cannot access free variable 'e'`. This is a Python 3 subtlety that trips up even experienced developers. I caught it in testing and directed the fix — capture `str(e)` into a local variable before the lambda.

## The Output: What Claude Built

The final application:

| Component | Lines | Purpose |
|-----------|-------|---------|
| `podcast_gui.py` | 1,706 | Full GUI application |
| `podcast_transcriber.py` | 389 | CLI application |
| `database.py` | 276 | SQLite database layer |
| `DESIGN.md` | ~550 | System design document |
| `README.md` | ~220 | User-facing documentation |
| **Total** | **~2,400** | **Code** |

Features I got without writing a single line:
- Dual transcription engine support (Whisper + faster-whisper)
- Speaker diarization via pyannote.audio
- AI-powered transcript analysis via OpenAI GPT
- Thread-safe model caching with background preloading
- Cooperative cancellation at every pipeline stage
- Per-segment progress with live ETA estimation
- Dynamic page sizing with debounced resize handling
- Podcast artwork loading with center-crop and caching
- macOS native menu bar integration via CoreFoundation ctypes
- Markdown rendering in the analysis panel
- SQLite persistence with cascade deletes and schema migrations

[SCREENSHOT: Full application window with sidebar showing multiple podcasts with artwork, an episode list with various statuses, and the analysis panel open in the 3rd column]

## What I Learned

**The 20-year developer is still essential.** Claude never pushed back on a bad idea. If I'd asked for a single-threaded GUI that blocks during transcription, I would have gotten one. The quality of the output was directly proportional to the quality of my direction. Knowing what *should* exist — cancellation checkpoints, thread-safe caches, cooperative signaling — was the difference between a demo and an application.

**Iteration speed is transformative.** The feedback loop between "I want this" and "here it is" shrunk from hours to minutes. Not because the problems were simpler, but because the mechanical work of translating design decisions into code was instantaneous. I spent my time on *what* and *why*, not *how*.

**Bugs still happen, and experience still catches them.** The exception variable scoping bug. Thread safety issues. Missing cursor propagation in tkinter. These are the kinds of things you learn from years of getting bitten. Claude wrote code that was generally correct, but "generally correct" is not "correct."

**Documentation came free.** The README and DESIGN.md were generated alongside the code. When I changed the code, the documentation updated automatically. This alone would justify the approach for any project — documentation that stays in sync because it's generated by the same process that writes the code.

**I still don't know if this is "programming."** I didn't write code. But I made every architectural decision, caught bugs, chose tradeoffs, enforced consistency, and steered the design. If programming is "translating intent into working software," then yes, I programmed. If programming is "writing code," then no, I didn't. I suspect this distinction will matter less and less over time.

## Would I Do It Again?

I'm already doing it again. Rangoli is my daily podcast tool now. When I want a new feature, I describe it, review the implementation, and ship it. The app works. The code is clean. The architecture is sound.

The most honest thing I can say is this: I built exactly the application I would have built myself, in a fraction of the time, and I enjoyed the process more. I spent my time on the interesting parts — the design, the tradeoffs, the user experience — and skipped the parts that feel like transcription.

Which, for an app about transcription, feels appropriate.

---

*Rangoli is open source under the MIT license. The entire codebase — every line — was written by Claude Code (Anthropic's AI coding agent), directed by a developer who kept his hands off the keyboard and his opinions very much on.*

[SCREENSHOT: The Rangoli application icon — a colorful rangoli pattern]
