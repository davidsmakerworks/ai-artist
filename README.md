# ai-artist
A.R.T.I.S.T. - Audio-Responsive Transformative Imagination Synthesis Technology

Generates images and verses of poetry based on user voice input. If no input is
received for some time, autonomously generates a new creation inspired by the
previous scene.

Uses OpenAI DALL-E 2 to generate images, GPT Chat Completion to generate verses
and Whisper API to transcribe speech to text.

Uses Azure Speech API to convert text to speech.

Uses Azure Blob Storage to store downloadable images.

Requries paid OpenAI API key, Azure Cognitive Services Speech Service (free tier)
and Azure Storage account.

This program is now esentially feature complete, but due to the addition of many
features, the code needs to be refactored to make things like error handling more
practical. For now, a brute-force approach of automatically restarting if there is 
an unhandled exception has been employed.

TODO: Major cleanup and refactor

TODO: Improve error handling for any/all web requests

TODO: Improve HTML template, add CSS

TODO: Add method to change microphone sensitivity/volume
