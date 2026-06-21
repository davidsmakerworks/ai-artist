# A.R.T.I.S.T. - Audio-Responsive Transformative Imagination Synthesis Technology

Generates images and verses of poetry based on user voice input. If no input is
received for some time, autonomously generates a new creation inspired by the
previous scene. A game controller can be used to trigger creations, navigate
recent works, and control the installation; hot-plug connect and reconnect are
supported.

## Technologies used
- OpenAI GPT Image 1, Stability AI SDXL, Stability AI Stable Image, or Fal.ai to generate images
- OpenAI GPT Chat Completion, Anthropic Claude, or OpenRouter (native SDK) to generate
verses and daydream image prompts, and to choose the best verse from multiple options if
the critic is enabled. Each AI character role (artist, poet, critic, visionary, emotion_chip,
raconteur, archivist) has its own independently configurable service, model, and
provider-specific options in `config.json`.
- OpenAI Whisper API to transcribe speech.
- Azure Speech API or OpenRouter TTS to convert text to speech (selectable via `speech_service` in `config.json`)
- Azure Blob Storage to store downloadable images
- Claude Code to assist with refactoring and feature implementation

## API keys/services required
- Paid OpenAI API key (Whisper transcription; also required for GPT Image 1 or OpenAI chat characters)
- Azure Storage account (`AZURE_STORAGE_KEY`)
- Azure Cognitive Services Speech Service (`AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`) — required only when `speech_service` is `"azure"` (the default)
- Paid Stability AI API key if using SDXL or Stable Image
- Paid Anthropic API key if any character uses `"service": "anthropic"`
- Paid OpenRouter API key if any character uses `"service": "openrouter"`, or when `speech_service` is `"openrouter"`
- Paid Fal.ai API key if using Fal.ai image models (e.g. `image_model: "fal-ai/flux/schnell"`)

A Python virtual environment (venv) should be created to avoid conflicts with system packages.

## Possible future enhancements
- Improve error handling for any/all web requests
- Add method to change microphone sensitivity/volume
- Handle slow uploads to Azure Blob Storage
- Allow authenticated web access to recent creations
