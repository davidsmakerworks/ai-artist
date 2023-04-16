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

import array

import pyaudio

from typing import Tuple


class AudioPlayer:
    """
    Audio player class for playing audio streams.

    TODO: Troubleshoot popping/clicking sounds when playing audio
    """

    def __init__(
        self, sample_width: int = 2, channels: int = 2, rate: int = 44000
    ) -> None:
        self._pyaudio = pyaudio.PyAudio()

        self.sample_width = sample_width
        self.channels = channels
        self.rate = rate

    @property
    def sample_width(self) -> int:
        return pyaudio.get_sample_size(self._audio_format)

    @sample_width.setter
    def sample_width(self, sample_width: int) -> None:
        self._audio_format = pyaudio.get_format_from_width(sample_width)

    def play(self, audio_stream: bytes) -> None:
        stream = self._pyaudio.open(
            format=self._audio_format,
            channels=self.channels,
            rate=self.rate,
            output=True,
        )

        stream.write(audio_stream)
        stream.stop_stream()
        stream.close()

    def terminate(self) -> None:
        self._pyaudio.terminate()


class AudioRecorder:
    """
    Audio recorder class for recording audio streams.
    """

    def __init__(self, sample_width: int = 2, channels: int = 2, rate: int = 44000) -> None:
        self._pyaudio = pyaudio.PyAudio()

        self.sample_width = sample_width
        self.channels = channels
        self.rate = rate

        self._audio_format = pyaudio.get_format_from_width(self.sample_width)

    def record(
        self,
        max_duration: int,
        chunk_size: int = 1024,
        silence_threshold: int = 2000,
        min_frames: int = 18,
        max_silent_frames: int = 10,
    ) -> Tuple[bytes, bool]:
        """
        Record audio for up to max_duration seconds.

        Parameters:
            max_duration (int): Maximum duration of audio to record in seconds
            chunk_size (int): Size of audio chunks to read from audio stream
            silence_threshold (int): Threshold for detecting silence
            min_frames (int): Minimum number of non-silent frames required for valid audio
            max_silent_frames (int): Number of silent frames to wait before ending recording

        Returns:
            bytes: Audio stream
            bool: True if valid audio was recorded, False otherwise

        TODO: Improve silence detection

        TODO: Trim pre-audio silence
        """

        stream = self._pyaudio.open(
            format=self._audio_format, channels=self.channels, rate=self.rate, input=True
        )

        frames = []
        num_frames = 0
        silent_frames = 0
        silence_detected = False
        was_silent = True

        max_frames = int(max_duration * self.rate / chunk_size)

        while (num_frames < max_frames) and not silence_detected:
            num_frames += 1
            data = stream.read(chunk_size)
            data_array = array.array("h", data)

            max_value = max(data_array)

            if max_value < silence_threshold:
                if was_silent:
                    silent_frames += 1
                was_silent = True
            else:
                silent_frames = 0
                was_silent = False

            frames.append(data)

            if silent_frames > max_silent_frames:
                silence_detected = True

        if num_frames < min_frames:
            valid_audio = False
        else:
            valid_audio = True

        stream.stop_stream()
        stream.close()

        return (b"".join(frames[:-max_silent_frames]), valid_audio)

    def terminate(self) -> None:
        self._pyaudio.terminate()
