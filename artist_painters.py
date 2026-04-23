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

from openai import OpenAI

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


class StableImageCreator:
    def __init__(self, api_key: str, service: str, sd3_model: str | None = None) -> None:
        """
        Initialize the StableImageCreator object.

        Args:
            api_key (str): The API key for the Stability AI API.
            service (str): The service to use for image generation.
            sd3_model (str): The SD3 model to use for image generation if applicable
        """
        self.api_key = api_key
        self.service = service
        self.sd3_model = sd3_model

    def generate_image_data(self, prompt: str) -> bytes:
        # Check this here instead of in initializer to allow for
        # dynamic model switching
        if self.service not in ["core", "ultra", "sd3"]:
            raise ValueError(f"Invalid Stable Image service specified: {self.service}")

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "accept": "image/*",
            "stability-client-id": "A.R.T.I.S.T.",
        }

        files = {"none": ""}

        data = {
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "output_format": "png",
        }

        if self.sd3_model:
            data["model"] = self.sd3_model

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/{self.service}",
            headers=headers,
            files=files,
            data=data,
        )

        if response.status_code == 200:
            return response.content
        elif response.status_code == 403:
            logger.error("Content filter triggered")
            raise RuntimeError("Content filter triggered")
        else:
            raise RuntimeError(f"Stable Image model error: {str(response.json())}")


class SDXLCreator:
    def __init__(
        self,
        api_key: str,
        img_width: int,
        img_height: int,
        steps: int,
        cfg_scale: float,
    ) -> None:
        self.api_key = api_key
        self.img_width = img_width
        self.img_height = img_height
        self.steps = steps
        self.cfg_scale = cfg_scale

    def generate_image_data(self, prompt: str) -> bytes | None:
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "image/png",
            },
            json={
                "text_prompts": [{"text": prompt}],
                "width": self.img_width,
                "height": self.img_height,
                "steps": self.steps,
                "cfg_scale": self.cfg_scale,
            },
        )

        if response.status_code == 200:
            if response.headers["Finish-Reason"] == "CONTENT_FILTERED":
                logger.error("Content filter triggered")
                raise RuntimeError("Content filter triggered")
            elif response.headers["Finish-Reason"] == "ERROR":
                raise RuntimeError("Error generating image")
            elif response.headers["Content-Type"] == "image/png":
                return response.content
            else:
                raise RuntimeError("No image data returned")
        else:
            raise RuntimeError(f"Stable Diffusion XL model error: {str(response.json())}")


class DallE2Creator:
    def __init__(self, api_key: str, img_width: int, img_height: int) -> None:
        self.api_key = api_key
        self.img_width = img_width
        self.img_height = img_height

        self._openai_client = OpenAI()
        self._openai_client.api_key = api_key

    def generate_image_data(self, prompt: str) -> bytes:
        img_size = f"{self.img_width}x{self.img_height}"

        try:
            response = self._openai_client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size=img_size,
                response_format="b64_json",
                user="A.R.T.I.S.T.",
            )
        except Exception as e:
            logger.error(f"Image creation response: {response}")
            logger.exception(e)
            raise

        return base64.b64decode(response.data[0].b64_json)


class DallE3Creator:
    def __init__(
        self, api_key: str, img_width: int, img_height: int, quality: str = "standard"
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
                model="dall-e-3",
                prompt=prompt,
                size=img_size,
                quality=self.quality,
                response_format="b64_json",
                user="A.R.T.I.S.T.",
            )
        except Exception as e:
            logger.error(f"Image creation response: {response}")
            logger.exception(e)
            raise

        return base64.b64decode(response.data[0].b64_json)


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
