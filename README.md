# A.R.T.I.S.T. - Audio-Responsive Transformative Imagination Synthesis Technology

Generates images and verses of poetry based on user voice input. If no input is
received for some time, autonomously generates a new creation inspired by the
previous scene.

## Technologies used
- OpenAI DALL-E 2, OpenAI DALL-E 3, Stability AI SDXL or Stability AI Stable Image to generate images
- OpenAI GPT Chat Completion to generate verses and Whisper API to
transcribe speech to text
- Azure Speech API to convert text to speech
- Azure Blob Storage to store downloadable images.

## API keys/services required
- Paid OpenAI API key
- Azure Cognitive Services Speech Service (free tier)
- Azure Storage account
- Paid Stability AI API key if using SDXL or Stable Image

The code needs additional refactoring to make things like error handling more
practical. For now, a brute-force approach of automatically restarting if there is 
an unhandled exception has been employed.

For Debian/Ubuntu, you may need to manually compile and install OpenSSL 1.1 because
the Azure Cognitive Services Speech SDK does not yet support OpenSSL 3.0. Instructions
for installation are [here](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?tabs=linux%2Cubuntu%2Cdotnetcli%2Cdotnet%2Cjre%2Cmaven%2Cnodejs%2Cmac%2Cpypi&pivots=programming-language-python).

A Python vritual environment (venv) should be created to avoid conflicts with system packages.

## Possible future enhancements
- Cleanup and refactor
- Improve error handling for any/all web requests
- Add method to change microphone sensitivity/volume
- Handle slow uploads to Azure Blob Storage
- Improve handling of new OpenAI API response format
- Allow authenticated web access to recent creations
- Track recent daydream topics to avoid getting stuck on particular subjects (jellyfish, mermaids, etc.)
- Add support for Stable Image Core presets in place of prompt prefixes when using Core model
- Implement support for different image models for user prompts and daydreams (i.e., to use better model
for user prompts and more economical model for daydreams)
- Implement automatic resizing of returned images that don't fit the desired size
