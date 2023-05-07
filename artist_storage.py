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

from azure.storage.blob import BlobServiceClient, ContentSettings

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


class ArtistStorage:
    def __init__(
        self, storage_key: str, storage_account: str, storage_container: str
    ) -> None:
        self._blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account}.blob.core.windows.net",
            credential=storage_key,
        )
        self._blob_container_client = self._blob_service_client.get_container_client(
            container=storage_container
        )

    def upload_blob(self, blob_name: str, data: bytes, content_type: str) -> None:
        content_settings = ContentSettings(content_type=content_type)
        
        self._blob_container_client.upload_blob(
            name=blob_name,
            data=data,
            overwrite=True,
            content_settings=content_settings,
        )