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
from anthropic.types import TextBlock
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openrouter import OpenRouter

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


class OpenAIChatCharacter:
    def __init__(
        self,
        system_prompt: str,
        model: str,
        api_key: str,
        provider_options: dict | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._provider_options = provider_options or {}

        self._openai_client = OpenAI()
        self._openai_client.api_key = api_key

    def get_chat_response(self, message: str) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": message},
        ]

        response = self._openai_client.chat.completions.create(
            model=self._model,
            messages=messages,
            **self._provider_options,
        )

        return response.choices[0].message.content or ""


class ClaudeChatCharacter:
    def __init__(
        self,
        system_prompt: str,
        model: str,
        api_key: str,
        provider_options: dict | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._provider_options = provider_options or {}

        self._anthropic_client = Anthropic(api_key=api_key)

    def get_chat_response(self, message: str) -> str:
        response = self._anthropic_client.messages.create(
            max_tokens=1024,
            model=self._model,
            system=self._system_prompt,
            messages=[{"role": "user", "content": message}],
            **self._provider_options,
        )

        content = response.content[0]
        assert isinstance(content, TextBlock)
        return content.text


class OpenRouterChatCharacter:
    def __init__(
        self,
        system_prompt: str,
        model: str,
        api_key: str,
        provider_options: dict | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._provider_options = provider_options or {}

        self._client = OpenRouter(api_key=api_key)

    def get_chat_response(self, message: str) -> str:
        response = self._client.chat.send(
            model=self._model,
            x_open_router_title="A.R.T.I.S.T.",
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": message},
            ],
            **self._provider_options,
        )

        return str(response.choices[0].message.content)
