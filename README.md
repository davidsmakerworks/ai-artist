# A.R.T.I.S.T. - Audio-Responsive Transformative Imagination Synthesis Technology

Generates images and verses of poetry based on user voice input. If no input is
received for some time, autonomously generates a new creation inspired by the
previous scene. A game controller can be used to trigger creations, navigate
recent works, and control the installation; hot-plug connect and reconnect are
supported.

## Technologies used
- OpenAI GPT Image 1 or Fal.ai to generate images
- OpenAI GPT Chat Completion, Anthropic Claude, or OpenRouter (native SDK) to generate
verses and daydream image prompts, and to choose the best verse from multiple options if
the critic is enabled. Each AI character role (artist, poet, critic, visionary, emotion_chip,
raconteur, archivist) has its own independently configurable service, model, and
provider-specific options in `config.json`.
- OpenAI Whisper API to transcribe speech.
- OpenAI Moderation API to screen each verse and image prompt before an image is generated.
- Azure Speech API or OpenRouter TTS to convert text to speech (selectable via `speech_service` in `config.json`)
- Azure Blob Storage to store downloadable images
- Claude Code to assist with refactoring and feature implementation

## API keys/services required
- Paid OpenAI API key (`OPENAI_API_KEY`) — always required for Whisper transcription and moderation; also used for GPT Image 1 and OpenAI chat characters
- Azure Storage account (`AZURE_STORAGE_KEY`)
- Azure Cognitive Services Speech Service (`AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`) — required only when `speech_service` is `"azure"` (the default)
- Paid Anthropic API key (`ANTHROPIC_API_KEY`) if any character uses `"service": "anthropic"`
- Paid OpenRouter API key (`OPENROUTER_API_KEY`) if any character uses `"service": "openrouter"`, or when `speech_service` is `"openrouter"`
- Paid Fal.ai API key (`FAL_API_KEY`) if using Fal.ai image models (e.g. `image_model: "fal-ai/flux/schnell"`)

Keys are read from the environment. A `.env` file next to `main.py` is loaded automatically if present (or pass `--env-file <path>`); values in the file override the shell environment.

A Python virtual environment (venv) should be created to avoid conflicts with system packages.

## Running

```
python main.py [--config config.json] [--local-config config-local.json] [--env-file .env]
```

`artist.sh` runs `main.py` with the interpreter from `venv/` on Linux.

## Content safety

`sfw_mode` (default `true`) appends `sfw_safety_clause` to every character's system prompt; startup fails if `sfw_mode` is on and the clause is empty. Independently of `sfw_mode`, every verse and final image prompt is checked with the OpenAI Moderation API, and a flagged result aborts the creation.

## Local configuration overrides

A `config-local.json` file placed alongside `config.json` (or any path passed via `--local-config`) is deep-merged on top of the base config before the application starts. Use it for machine-specific settings (model choices, voices, paths) without modifying the checked-in `config.json`. Character sub-objects are merged recursively, so you can override a single field such as `"model"` inside `"poet"` without repeating the entire character block. Unknown keys are logged as warnings and ignored. `config-local.json` is listed in `.gitignore`.

## Possible future enhancements
- Improve error handling for any/all web requests
- Use a single shared OpenAI client instead of one per class
- Add method to change microphone sensitivity/volume
- Handle slow uploads to Azure Blob Storage
- Allow authenticated web access to recent creations
