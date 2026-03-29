# A.R.T.I.S.T. — Developer Tips

## How the Pieces Fit Together

The system has a clear separation between three layers:

**Service wrappers** (`openai_tools.py`, `anthropic_tools.py`, `artist_speech.py`, `artist_storage.py`, `artist_moderator.py`, `audio_tools.py`) — thin, focused classes that talk to one external API or device. These generally have no business logic.

**Display / creation classes** (`artist_classes.py`) — `ArtistCanvas`, `StatusScreen`, `ArtistCreation`, and all the image-creator classes live here. Image generation (network calls) is bundled in the same file as rendering (pygame). This is worth noting if you want to reorganize.

**Application logic** (`main.py`) — everything else. `AppConfig`, `AppState`, all pipeline functions, the main loop, and the two factory functions (`create_painter`, `create_chat_character`). At ~1850 lines it's the heaviest file.

The two factories are the single point where you'd add a new image model or LLM backend. Everything above them in the call stack is model-agnostic.

---

## The OpenAI vs. Anthropic Character Difference

This is the most subtle behavioral split in the codebase.

`ChatCharacter` (OpenAI, `openai_tools.py`) accumulates a full message history — it's genuinely multi-turn. After `get_chat_response()` the assistant reply is appended to `self._messages`. Calling `reset()` wipes it back to just the system prompt.

`ClaudeChatCharacter` (Anthropic, `anthropic_tools.py`) does **not** append the assistant response after a call. `self._messages` only accumulates user turns. In practice this doesn't matter because every character (poet, critic, visionary, artist) calls `.reset()` before each use, but the `ai_artist` character was presumably intended to be multi-turn. If you switch `chat_service` from `openai` to `anthropic`, the artist loses its history between calls. This is a latent inconsistency.

Also: `ClaudeChatCharacter` has a hardcoded `max_tokens=1024`. Very long verses or visionary prompts could be silently truncated.

---

## The Critic's Fragile Number Parser

`get_best_verse()` in [main.py:276](main.py#L276) determines the winning poem by scanning the critic's response for the first digit character:

```python
for c in critic_verdict:
    if c.isdigit():
        chosen_poem = int(c)
        break
```

This works well when the critic responds "Poem 1 is the best choice." It breaks silently if the critic says "the 3rd poem" (returns 3, which may not be a valid index) or if the model starts its response with a year or any other number. Falls back to `random.choice(verses)` if no digit is found. If you enable the critic, instruct it in its system prompt to reply with only a single digit.

---

## Only Square Images Supported

`load_config()` hard-rejects any config where `img_width != img_height` ([main.py:816](main.py#L816)). This affects DALL-E 3 (which supports 1792×1024 and 1024×1792) and Stable Image. The code to resize a non-square image already exists in `render_creation_display()`, so lifting this restriction is primarily a config-validation change.

---

## Multiple OpenAI Clients

Each of `DallE2Creator`, `DallE3Creator`, `GptImage1Creator`, `Transcriber`, and every `ChatCharacter` instance creates its own `OpenAI()` client. On a typical run you'll have 4–6 independent clients. The README flags this as a known issue. It's harmless but wasteful — consolidating them would also simplify credential management.

---

## SSML Voice Features Are Wired Up but Unused

`ArtistSpeech` ([artist_speech.py:58](artist_speech.py#L58)) stores `style`, `pitch`, `rate`, and `role` as instance attributes. The SSML generator includes them if non-None. But nothing in the application ever sets them after initialization — they remain `None` for every call. If you want to give the installation a more expressive voice (whispery for daydreams, excited for user creations) the infrastructure is already there.

---

## Speech Caching Key

The cache key is SHA256 of `language + gender + voice + text` ([artist_speech.py:133](artist_speech.py#L133)). If you change the voice or language in `config.json`, all cached files become unreachable (they won't be replayed, new files will be generated). The old files aren't deleted. You'll want to clear `cache/` after a voice change to avoid stale accumulation.

---

## Debug Log Path Is Hardcoded

`wait_for_action()` calls `get_debug_log_surface(log_file="artist.log", ...)` ([main.py:1047](main.py#L1047)) with a literal path, not from config. If `log_config.py` is ever changed to write the log elsewhere, the on-screen debug view will silently show an error rather than failing obviously.

---

## Recents Are Screenshot-Based

When you browse recents with LEFT/RIGHT, the code loads the saved PNG screenshot from `output/` ([main.py:1071](main.py#L1071)) — the full canvas (image + verse baked together). The raw generated image is not stored separately. If you want to show the same creation at a different layout or aspect ratio later, it isn't possible from stored data.

---

## Emotion Chip Trigger Logic

Emotional state is updated in two ways:
- **After a user creation** — `generate_emotional_state()` runs, generating a fresh state from the last N user prompts.
- **After every `emotion_drift_interval` daydreams** — `drift_emotional_state()` runs, evolving the state based on recent daydream themes.

The counter `state.daydreams_since_user_prompt` resets to 0 on any user creation. So if users interact frequently the emotional state is purely user-driven; the drift path only activates during long idle periods.

Emotion is injected into both the poet's base prompt and the artist's daydream prompt, but not into the visionary's image prompt.

---

## Daydream Rate Limiting

Manual daydreams (pressing `d`) are rate-limited by a sliding window: `manual_daydream_window` seconds, `manual_daydream_limit` triggers ([main.py:973](main.py#L973)). Expired timestamps are cleaned up at the top of the `wait_for_action()` loop. Auto-daydreams are not rate-limited in the same way — they're controlled purely by the scheduled `next_change_time`.

---

## Audio Silence Detection Known Limitations

The recorder (`audio_tools.py:88`) has two TODOs in its docstring:
- Improve silence detection (currently just peak amplitude vs. threshold)
- Trim pre-audio silence

The current approach: if peak amplitude in a chunk is below `silence_threshold` (default 2000), it counts as a silent frame. After `max_silent_frames` (default 10) consecutive silent chunks, recording stops. The silence threshold is not configurable from `config.json` — it's a hardcoded default on the `record()` call in `capture_user_audio()`. To tune microphone sensitivity you need to either change the default in `audio_tools.py` or pass it explicitly from the config.

---

## Possible Enhancements

### From the README Backlog

- **Single shared OpenAI client** — pass one `OpenAI()` instance through to all consumers rather than creating one per class.
- **Better API error handling** — Stability AI and Azure uploads currently log and continue on failure; transient failures could benefit from simple retry with backoff.
- **Configurable microphone sensitivity** — expose `silence_threshold` and `max_silent_frames` in `config.json` and pass them to `audio_recorder.record()`.
- **Handle slow Azure uploads** — currently upload is synchronous and blocks before the display is shown. Running it in a background thread would let the creation appear on screen immediately.
- **Authenticated web access** — the HTML page uploaded to Azure is currently public. Add SAS token generation or a simple auth layer for private installations.

### Ideas from Code Review

**Per-character LLM backend** — the single `chat_service` key forces all characters to use the same provider. A dict like `{ "poet": "anthropic", "visionary": "openai" }` would let you use cheaper or faster models for simpler roles (e.g. critic) while using a stronger model for the poet.

**Non-blocking pipeline** — the entire creation pipeline runs on the main thread, blocking input for 10–30+ seconds. A `threading.Thread` for the pipeline would allow ESC to abort a running generation. The main challenge is thread-safe pygame surface updates.

**Better critic output format** — use the model's structured output / JSON mode (available on both OpenAI and recent Anthropic models) to get the critic to return `{"choice": 2}` rather than scanning free text for a digit.

**Configurable `max_tokens` for Claude** — expose `max_tokens` in config or per-character, rather than the hardcoded 1024 in `ClaudeChatCharacter`.

**SSML voice styles** — use the style/pitch/rate fields already wired up in `ArtistSpeech` to differentiate daydream speech from user-response speech.

**Non-square image support** — remove the `img_width == img_height` constraint in `load_config()`, and update `ArtistCanvas.render_creation()` to lay out non-square images correctly. The resize fallback is already in place.

**Configurable audio device** — PyAudio selects the system default for both input and output. Adding device index params to `AudioRecorder` and `AudioPlayer` (and exposing them in config) would help multi-sound-card setups.

**Slideshow / idle gallery mode** — a configurable option to auto-cycle through recents while idle, rather than showing a static status screen.

**Verse max length control** — add a config key for max verse lines or word count. The poet system prompt influences this, but prompt engineering alone is unreliable. A post-process trim or a function-call constraint would be more robust.

**Log rotation** — `artist.log` grows unbounded. Python's `RotatingFileHandler` is a one-line change in `log_config.py`.
