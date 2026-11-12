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

import base64
import logging
import requests

from fal_client import SyncClient as FalSyncClient
from openai import OpenAI

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


class GptImage1Creator:
    def __init__(
        self, api_key: str, img_width: int, img_height: int, quality: str = "medium"
    ) -> None:
        self.api_key = api_key
        self.img_width = img_width
        self.img_height = img_height
        self.quality = quality

        self._openai_client = OpenAI()
        self._openai_client.api_key = api_key

    def generate_image_data(self, prompt: str) -> bytes:
        img_size = f"{self.img_width}x{self.img_height}"

        try:
            response = self._openai_client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=img_size,
                quality=self.quality,
                background="opaque",
                user="A.R.T.I.S.T.",
            )
        except Exception as e:
            logger.error(f"Image creation response: {response}")
            logger.exception(e)
            raise

        return base64.b64decode(response.data[0].b64_json)


class FalImageCreator:
    def __init__(
        self, api_key: str, model: str, img_width: int, img_height: int
    ) -> None:
        self._client = FalSyncClient(key=api_key)
        self._model = model
        self._img_width = img_width
        self._img_height = img_height

    def generate_image_data(self, prompt: str) -> bytes:
        result = self._client.run(
            self._model,
            arguments={
                "prompt": prompt,
                "image_size": {"width": self._img_width, "height": self._img_height},
            },
        )

        image_url = result["images"][0]["url"]

        response = requests.get(image_url)
        response.raise_for_status()

        return response.content
