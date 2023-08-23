# ai-artist
A.R.T.I.S.T. - Audio-Responsive Transformative Imagination Synthesis Technology

Generates images and verses of poetry based on user voice input. If no input is
received for some time, autonomously generates a new creation inspired by the
previous scene.

Uses OpenAI DALL-E 2 or Stability AI SDXL to generate images.

Uses OpenAI GPT Chat Completion to generate verses and Whisper API to
transcribe speech to text.

Uses Azure Speech API to convert text to speech.

Uses Azure Blob Storage to store downloadable images.

Requries paid OpenAI API key, Azure Cognitive Services Speech Service (free tier)
and Azure Storage account. Also requires paid Stability AI API key if using SDXL.

The code needs additional refactoring to make things like error handling more
practical. For now, a brute-force approach of automatically restarting if there is 
an unhandled exception has been employed.

For Debian/Ubuntu, you may need to manually compile and install OpenSSL 1.1 because
the Azure Cognitive Services Speech SDK does not yet support OpenSSL 3.0. Instructions
for installation are [here](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/quickstarts/setup-platform?tabs=linux%2Cubuntu%2Cdotnetcli%2Cdotnet%2Cjre%2Cmaven%2Cnodejs%2Cmac%2Cpypi&pivots=programming-language-python).

TODO: Cleanup and refactor

TODO: Improve error handling for any/all web requests

TODO: Improve HTML template, add CSS

TODO: Add method to change microphone sensitivity/volume

TODO: Make SDXL options configurable

TODO: Improve SDXL filter handling