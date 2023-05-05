# MIT License

# Copyright (c) 2023 David Rice

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import io
import logging
import wave

import openai

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())

class Transcriber:
    def __init__(
        self, channels: int, sample_width: int, framerate: int, model: str
    ) -> None:
        self.channels = channels
        self.sample_width = sample_width
        self.framerate = framerate
        self.model = model

    def transcribe(self, audio_stream: bytes) -> str:
        """
        Transcribe audio stream to text.
        """
        audio_data = io.BytesIO()
        writer = wave.open(audio_data, "wb")

        writer.setnchannels(self.channels)
        writer.setsampwidth(self.sample_width)
        writer.setframerate(self.framerate)

        writer.writeframes(audio_stream)

        writer.close()
        
        audio_data.seek(0)
        audio_data.name = "audio.wav" # Name hint only, not a file on disk

        try:
            response = openai.Audio.transcribe(model=self.model, file=audio_data)
        except Exception as e:
            logger.error(f"Transcriber response: {response}")
            logger.exception(e)
            raise

        return response["text"]


class ChatResponse:
    def __init__(self, response: dict) -> None:
        self._response = response

    @property
    def content(self) -> str:
        return self._response["choices"][0]["message"]["content"]

    @property
    def total_tokens_used(self) -> int:
        return self._response["usage"]["total_tokens"]


class ChatCharacter:
    def __init__(self, system_prompt: str, model: str) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self.reset()

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": self._system_prompt}]

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        if self._messages[0]["role"] == "system":
            self._messages[0]["content"] = prompt
        else:
            raise RuntimeError("Invalid structure of ChatCharacter._messages")

    def get_chat_response(self, message: str) -> ChatResponse:
        self._messages.append({"role": "user", "content": message})

        response = openai.ChatCompletion.create(
            model=self._model, messages=self._messages
        )

        self._messages.append(response["choices"][0]["message"])

        return ChatResponse(response)
