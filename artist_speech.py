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

import hashlib
import io
import logging
import os
import wave

import azure.cognitiveservices.speech as speechsdk

import xml.etree.ElementTree as ET

from audio_tools import AudioPlayer

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())

class ArtistSpeech:
    def __init__(
        self,
        subscription_key: str,
        region: str,
        language: str,
        gender: str,
        voice: str,
        cache_dir: str,
        output_format: speechsdk.SpeechSynthesisOutputFormat = speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
        channels: int = 1,
        sample_rate: int = 16000,
        sample_width: int = 2,
    ) -> None:
        self._cache_dir = cache_dir

        self.language = language
        self.gender = gender
        self.voice = voice

        self.style = None
        self.pitch = None
        self.rate = None
        self.role = None

        speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )
        speech_config.set_speech_synthesis_output_format(output_format)

        self._synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )

        self._player = AudioPlayer(
            sample_width=sample_width, channels=channels, rate=sample_rate
        )

    def _generate_ssml(self, text: str) -> str:
        speak_attrib = {
            "version": "1.0",
            "xmlns": "http://www.w3.org/2001/10/synthesis",
            "xmlns:mstts": "https://www.w3.org/2001/mstts",
            "xml:lang": self.language,
        }

        ssml_root = ET.Element("speak", speak_attrib)

        ssml_subelements = []

        ssml_subelements.append({"voice": {"name": self.voice}})

        express_as_attrib = {}

        if self.style:
            express_as_attrib["style"] = self.style

        if self.role:
            express_as_attrib["role"] = self.role

        if express_as_attrib:
            ssml_subelements.append({"mstts:express-as": express_as_attrib})

        prosody_attrib = {}

        if self.pitch:
            prosody_attrib["pitch"] = self.pitch

        if self.rate:
            prosody_attrib["rate"] = self.rate

        if prosody_attrib:
            ssml_subelements.append({"prosody": prosody_attrib})

        ssml_parent = ssml_root

        for element in ssml_subelements:
            for tag, attrib in element.items():
                ssml_parent = ET.SubElement(ssml_parent, tag, attrib)

        ssml_parent.text = text

        return ET.tostring(ssml_root, encoding="unicode")

    def speak_text(self, text: str, use_cache: bool = True) -> None:
        if use_cache:
            text_details = self.language + self.gender + self.voice + text
            text_details_hash = hashlib.sha256(text_details.encode("utf-8")).hexdigest()

            cached_file_path = os.path.join(self._cache_dir, text_details_hash + ".wav")

            if not os.path.exists(cached_file_path):
                synthesis_result = self._synthesizer.speak_ssml(
                    self._generate_ssml(text)
                )

                with open(cached_file_path, "wb") as cached_file:
                    cached_file.write(synthesis_result.audio_data)

            with wave.open(cached_file_path, "rb") as cached_file:
                self._player.play(cached_file.readframes(cached_file.getnframes()))
        else:
            synthesis_result = self._synthesizer.speak_ssml(self._generate_ssml(text))
            audio_data = synthesis_result.audio_data

            wav_data = io.BytesIO(audio_data)

            with wave.open(wav_data, "rb") as virtual_file:
                self._player.play(virtual_file.readframes(virtual_file.getnframes()))
