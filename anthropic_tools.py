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


import logging

from anthropic import Anthropic

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


class ClaudeChatResponse:
    def __init__(self, response) -> None:
        self._response = response

    @property
    def content(self) -> str:
        return self._response.content[0].text

    @property
    def total_tokens_used(self) -> int:
        return self._response.usage.input_tokens + self._response.usage.output_tokens


class ClaudeChatCharacter:
    def __init__(
        self,
        system_prompt: str,
        model: str,
        api_key: str,
        temperature: float = 1.0,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._temperature = temperature

        self._anthropic_client = Anthropic(api_key=api_key)

        self.reset()

    def reset(self) -> None:
        self._messages = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def get_chat_response(self, message: str) -> ClaudeChatResponse:
        self._messages.append({"role": "user", "content": message})

        response = self._anthropic_client.messages.create(
            max_tokens=1024,
            model=self._model,
            temperature=self._temperature,
            system=self._system_prompt,
            messages=self._messages,
        )

        return ClaudeChatResponse(response)
