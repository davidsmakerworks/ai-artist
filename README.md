# A.R.T.I.S.T. - Audio-Responsive Transformative Imagination Synthesis Technology

Generates images and verses of poetry based on user voice input. If no input is
received for some time, autonomously generates a new creation inspired by the
previous scene.

## Technologies used
- OpenAI DALL-E 2, OpenAI DALL-E 3, Stability AI SDXL or Stability AI Stable 
Image to generate images
- OpenAI GPT Chat Completion or Anthropic Claude to generate verses and
daydream image prompts, and to choose the best verse from multiple options if
the critic is enabled.
- OpenAI Whisper API to transcribe speech.
- Azure Speech API to convert text to speech
- Azure Blob Storage to store downloadable images
- Claude Code to assist with refactoring and feature implementation

## API keys/services required
- Paid OpenAI API key
- Azure Cognitive Services Speech Service (free tier)
- Azure Storage account
- Paid Stability AI API key if using SDXL or Stable Image
- Paid Anthropic API key if using Claude

The code needs additional refactoring to make things like error handling more
practical. For now, a brute-force approach of automatically restarting if there is 
an unhandled exception has been employed.

For Debian/Ubuntu, you may need to manually compile and install OpenSSL 1.1 because
the Azure Cognitive Services Speech SDK does not yet support OpenSSL 3.0. Instructions
for installation are [here](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?tabs=linux%2Cubuntu%2Cdotnetcli%2Cdotnet%2Cjre%2Cmaven%2Cnodejs%2Cmac%2Cpypi&pivots=programming-language-python).

A Python virtual environment (venv) should be created to avoid conflicts with system packages.

## Possible future enhancements
- Cleanup and refactor (in progress - AI assisted)
- Use one OpenAI SDK client for entire application(?)
- Improve error handling for any/all web requests
- Add method to change microphone sensitivity/volume
- Handle slow uploads to Azure Blob Storage
- Improve handling of new OpenAI API response format
- Allow authenticated web access to recent creations
- Track recent daydream topics to avoid getting stuck on particular subjects (jellyfish, mermaids, etc.)
- Use AI to optimize image prompt based on type of prompt the model expects
