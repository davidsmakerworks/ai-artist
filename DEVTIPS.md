# A.R.T.I.S.T. — Developer Tips

## How the Pieces Fit Together

The system has a clear separation between four layers:

**Configuration & state** (`artist_config.py`) — `AppConfig`, `AppState`, `ButtonConfig`, `UserAction`, and `load_config()`. No pygame, no network calls. The single place to look when a config key is missing or a field type is wrong. `AppState.js` holds the active `pygame.joystick.Joystick` instance (or `None`) and is updated directly by `check_for_event()` when the controller connects or disconnects.

**Chat characters** (`artist_characters.py`) — `OpenAIChatCharacter`, `ClaudeChatCharacter`, and `OpenRouterChatCharacter`, plus their response wrapper classes. All three share the same single-turn interface: `get_chat_response(message) -> response`. The vendor-specific files (`openai_tools.py`, `anthropic_tools.py`, `openrouter_tools.py`) are now empty stubs reserved for future non-character utilities.

**Service wrappers** (`artist_speech.py`, `artist_storage.py`, `artist_moderator.py`, `audio_tools.py`) — thin, focused classes that talk to one external API or device. These generally have no business logic. `audio_tools.py` also contains `Transcriber`, which wraps the OpenAI Whisper API for speech-to-text.

**Display & image generation** — split across two files:
- `artist_classes.py` — pygame canvas and surface rendering: `ArtistCanvas`, `StatusScreen`, `ArtistCreation`, and the module-level surface functions (`get_prompt_surface`, `get_emotional_state_surface`, `get_debug_log_surface`, `draw_hourglass_indicator`, `show_status_screen`).
- `artist_painters.py` — image generator classes that make network calls: `StableImageCreator`, `SDXLCreator`, `GptImage1Creator`, `FalImageCreator`. All share a `generate_image_data(prompt) -> bytes` interface.

**Application logic** (`main.py`) — the creation pipeline, event loop, factory functions (`create_painter`, `create_chat_character`), and recents persistence. At ~1545 lines it's still the largest file but now focused on orchestration.

The two factories are the single point where you'd add a new image model or LLM backend. Everything above them in the call stack is model-agnostic.

---

## Environment Variable Loading

API keys and other secrets are read from the environment. On startup, `main()` resolves which source to use, in priority order:

1. **`--env-file <path>`** — loads the specified file via `python-dotenv` with `override=True`, replacing any already-set variables.
2. **`.env` in the same directory as `main.py`** — loaded automatically if present and `--env-file` is not given.
3. **Existing shell environment** — used as-is when neither of the above applies.

`override=True` means the file always wins over the shell environment, which makes behavior predictable when both exist. If you want the shell to take precedence, you'd need to drop the override flag.

---

## The Three Chat Backends

Three backends are available via `chat_service` in `config.json`. All live in `artist_characters.py` and share an identical single-turn interface: construct with `(system_prompt, model, api_key)`, call `get_chat_response(message)`, read `.content` from the result.

`OpenAIChatCharacter` uses the OpenAI SDK (`openai.chat.completions.create`). Each call sends only the system prompt and the single user message — there is no accumulated history.

`ClaudeChatCharacter` uses the Anthropic SDK (`anthropic.messages.create`). Same single-turn contract. Note the hardcoded `max_tokens=1024` — very long verses or visionary prompts could be silently truncated.

`OpenRouterChatCharacter` uses the native OpenRouter SDK (`openrouter.chat.send`). The model string is passed through verbatim (e.g. `"deepseek/deepseek-v4-5"`), so any model available on OpenRouter can be used by changing `config.json` alone. Requires `OPENROUTER_API_KEY`.

---

## The Critic's Fragile Number Parser

`get_best_verse()` in [main.py:223](main.py#L223) determines the winning poem by scanning the critic's response for the first digit character:

```python
for c in critic_verdict:
    if c.isdigit():
        chosen_poem = int(c)
        break
```

This works well when the critic responds "Poem 1 is the best choice." It breaks silently if the critic says "the 3rd poem" (returns 3, which may not be a valid index) or if the model starts its response with a year or any other number. Falls back to `random.choice(verses)` if no digit is found. If you enable the critic, instruct it in its system prompt to reply with only a single digit.

---

## Only Square Images Supported

`load_config()` hard-rejects any config where `img_width != img_height` ([artist_config.py:360](artist_config.py#L360)). This affects Stable Image and GPT Image 1 (which support non-square sizes). The code to resize a non-square image already exists in `render_creation_display()`, so lifting this restriction is primarily a config-validation change.

---

## Multiple OpenAI Clients

`GptImage1Creator` (in `artist_painters.py`), `Transcriber` (in `audio_tools.py`), and every `OpenAIChatCharacter` instance each creates its own `OpenAI()` client. On a typical run you'll have 3–5 independent clients. `FalImageCreator` uses a `FalSyncClient` (not OpenAI) initialized with the `FAL_API_KEY` — it does not set the key as an environment variable. The README flags the multiple-client issue as known. It's harmless but wasteful — consolidating them would also simplify credential management.

---

## SSML Voice Features Are Wired Up but Unused

`ArtistSpeech` ([artist_speech.py:58](artist_speech.py#L58)) stores `style`, `pitch`, `rate`, and `role` as instance attributes. The SSML generator includes them if non-None. But nothing in the application ever sets them after initialization — they remain `None` for every call. If you want to give the installation a more expressive voice (whispery for daydreams, excited for user creations) the infrastructure is already there.

---

## Speech Caching Key

The cache key is SHA256 of `language + gender + voice + text` ([artist_speech.py:133](artist_speech.py#L133)). If you change the voice or language in `config.json`, all cached files become unreachable (they won't be replayed, new files will be generated). The old files aren't deleted. You'll want to clear `cache/` after a voice change to avoid stale accumulation.

---

## Debug Log Path Is Hardcoded

`wait_for_action()` calls `get_debug_log_surface(log_file="artist.log", ...)` ([main.py:440](main.py#L440)) with a literal path, not from config. If `log_config.py` is ever changed to write the log elsewhere, the on-screen debug view will silently show an error rather than failing obviously.

---

## Recents Are Screenshot-Based

When you browse recents with LEFT/RIGHT, the code loads the saved PNG screenshot from `output/` ([main.py:455](main.py#L455)) — the full canvas (image + verse baked together). The raw generated image is not stored separately. If you want to show the same creation at a different layout or aspect ratio later, it isn't possible from stored data.

---

## Emotion Chip Trigger Logic

Emotional state is updated in two ways:
- **After a user creation** — `generate_emotional_state()` runs, generating a fresh state from the last N user prompts.
- **After every `emotion_drift_interval` daydreams** — `drift_emotional_state()` runs, evolving the state based on recent daydream themes.

The counter `state.daydreams_since_user_prompt` resets to 0 on any user creation. So if users interact frequently the emotional state is purely user-driven; the drift path only activates during long idle periods.

Emotion is injected into both the poet's base prompt and the artist's daydream prompt, but not into the visionary's image prompt.

---

## Joystick Hot-Plug

The controller is not required at startup. `init_joystick()` sets `state.js` to the first connected device, or `None` if none is found. From that point on, `check_for_event()` owns all joystick lifecycle transitions:

- **`JOYDEVICEADDED`** — creates a new `Joystick(event.device_index)`, initializes it, and stores it in `state.js`. Fires for controllers plugged in at any time, including after startup. On some platforms it also fires for controllers already connected when the app starts; re-initializing is harmless.
- **`JOYDEVICEREMOVED`** — clears `state.js = None`. This prevents stale `JOYAXISMOTION` events (which may briefly linger after disconnect) from being acted on.

`state.js` is used inside `check_for_event()` rather than being threaded as a parameter through `wait_for_action()`. This keeps the joystick in `AppState` where it belongs as runtime state, and avoids the need for a mutable wrapper or a changed return type on the event functions.

`js.get_button()` is only called in direct response to a `JOYBUTTONDOWN` event, which cannot fire from a disconnected device, so no exception handling around those calls is needed.

---

## Daydream Rate Limiting

Manual daydreams (pressing `d`) are rate-limited by a sliding window: `manual_daydream_window` seconds, `manual_daydream_limit` triggers ([main.py:340](main.py#L340)). Expired timestamps are cleaned up at the top of the `wait_for_action()` loop. Auto-daydreams are not rate-limited in the same way — they're controlled purely by the scheduled `next_change_time`.

---

## Audio Silence Detection Known Limitations

The recorder (`audio_tools.py:88`) has two TODOs in its docstring:
- Improve silence detection (currently just peak amplitude vs. threshold)
- Trim pre-audio silence

The current approach: if peak amplitude in a chunk is below `silence_threshold` (default 2000), it counts as a silent frame. After `max_silent_frames` (default 10) consecutive silent chunks, recording stops. The silence threshold is not configurable from `config.json` — it's a hardcoded default on the `record()` call in `capture_user_audio()`. To tune microphone sensitivity you need to either change the default in `audio_tools.py` or pass it explicitly from the config.

---

## Output Directory Disk Space Management

`enforce_output_disk_space()` runs automatically after every creation (`save_creation_locally` → `update_recents_and_scheduling` → `enforce_output_disk_space`). The call order matters: recents are updated first so the protected set is always current when cleanup runs.

**Thresholds** (configured in `config.json`):
- `disk_space_warn_percentage` (default 10) — cleanup triggers when free space falls below this.
- `disk_space_target_percentage` (default 20) — cleanup stops once free space reaches this.

**Protected files** — any `.png` whose `base_name` appears in `state.recents` is never deleted, preventing crashes when the user browses recents with LEFT/RIGHT.

**Deletion order** — unprotected `.png` files in `output/` are deleted oldest-first by mtime. The function re-checks disk usage after each deletion and stops as soon as the target is met, avoiding over-deletion.

Uses `shutil.disk_usage()` for cross-platform compatibility (Windows and Linux).

If the output directory runs out of deletable files before the target is reached, a warning is logged but no exception is raised — the drive may have other large consumers outside `output/`.

---

## Possible Enhancements

### From the README Backlog

- **Single shared OpenAI client** — pass one `OpenAI()` instance through to all consumers rather than creating one per class.
- **Better API error handling** — Stability AI and Azure uploads currently log and continue on failure; transient failures could benefit from simple retry with backoff.
- **Configurable microphone sensitivity** — expose `silence_threshold` and `max_silent_frames` in `config.json` and pass them to `audio_recorder.record()`.
- **Handle slow Azure uploads** — currently upload is synchronous and blocks before the display is shown. Running it in a background thread would let the creation appear on screen immediately.
- **Authenticated web access** — the HTML page uploaded to Azure is currently public. Add SAS token generation or a simple auth layer for private installations.

### Ideas from Code Review

**Per-character LLM backend** — the single `chat_service` key forces all characters to use the same provider (OpenAI, Anthropic, or OpenRouter). A dict like `{ "poet": "anthropic", "visionary": "openrouter" }` would let you mix providers per role. Note that OpenRouter already lets you mix model tiers within a single `chat_service`, so this matters most when you want to mix native Anthropic/OpenAI APIs with OpenRouter.

**Non-blocking pipeline** — the entire creation pipeline runs on the main thread, blocking input for 10–30+ seconds. A `threading.Thread` for the pipeline would allow ESC to abort a running generation. The main challenge is thread-safe pygame surface updates.

**Better critic output format** — use the model's structured output / JSON mode (available on both OpenAI and recent Anthropic models) to get the critic to return `{"choice": 2}` rather than scanning free text for a digit.

**Configurable `max_tokens` for Claude** — expose `max_tokens` in config or per-character, rather than the hardcoded 1024 in `ClaudeChatCharacter`. OpenAI and OpenRouter don't have this cap.

**SSML voice styles** — use the style/pitch/rate fields already wired up in `ArtistSpeech` to differentiate daydream speech from user-response speech.

**Non-square image support** — remove the `img_width == img_height` constraint in `load_config()` (`artist_config.py`), and update `ArtistCanvas.render_creation()` to lay out non-square images correctly. The resize fallback is already in place.

**Configurable audio device** — PyAudio selects the system default for both input and output. Adding device index params to `AudioRecorder` and `AudioPlayer` (and exposing them in config) would help multi-sound-card setups.

**Slideshow / idle gallery mode** — a configurable option to auto-cycle through recents while idle, rather than showing a static status screen.

**Verse max length control** — add a config key for max verse lines or word count. The poet system prompt influences this, but prompt engineering alone is unreliable. A post-process trim or a function-call constraint would be more robust.

**Single shared OpenAI client** — `GptImage1Creator`, `Transcriber`, and `OpenAIChatCharacter` each instantiate their own `OpenAI()`. Consolidating would simplify credential management.
