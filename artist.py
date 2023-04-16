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

"""
A.R.T.I.S.T. - Audio-Responsive Transformative Imagination Synthesis Technology

Generates images and verses of poetry based on user voice input.

Uses OpenAI's DALL-E 2 to generate images, GPT-3.5 Chat to generate verses
and Whisper API to transcribe speech.

Uses Azure Speech API to convert text to speech.

TODO: General cleanup and structural improvements

TODO: Upload results to a site so users can download their creations
"""

import base64
import hashlib
import json
import random
import os
import string
import wave

import pyaudio
import pygame
import openai

from audio_tools import AudioPlayer, AudioRecorder
from azure_speech import AzureSpeech
from enum import IntEnum
from pygame.locals import *
from typing import Union


class Button(IntEnum):
    BTN_A = 0
    BTN_B = 1
    BTN_X = 2
    BTN_Y = 3
    BTN_LB = 4
    BTN_RB = 5
    BTN_BACK = 6
    BTN_START = 7
    BTN_LS = 8
    BTN_RS = 9


class Transcriber:
    def __init__(
        self, temp_dir: str, channels: int, sample_width: int, framerate: int
    ) -> None:
        self.temp_dir = temp_dir
        self.channels = channels
        self.sample_width = sample_width
        self.framerate = framerate

    def transcribe(self, audio_stream: bytes) -> str:
        """
        Transcribe audio stream to text.

        TODO: Find a way to do this in memory without temporary file
        """
        temp_file_name = os.path.join(self.temp_dir, "input_audio.wav")

        writer = wave.open(temp_file_name, "wb")

        writer.setnchannels(self.channels)
        writer.setsampwidth(self.sample_width)
        writer.setframerate(self.framerate)

        writer.writeframes(audio_stream)

        with open(temp_file_name, "rb") as f:
            response = openai.Audio.transcribe(model="whisper-1", file=f)

        return response["text"]


def init_display(width: int, height: int) -> pygame.Surface:
    """
    Initialize pygame display.
    """
    pygame.init()

    pygame.mouse.set_visible(False)

    surface = pygame.display.set_mode((width, height), pygame.FULLSCREEN)

    surface.fill(pygame.Color("black"))

    pygame.display.update()

    return surface


def init_joystick() -> None:
    """
    Initialize joystick if one is connected.
    """
    pygame.joystick.init()

    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()


def speak_text(
    text: str, cache_dir: str, player: AudioPlayer, speech_svc: AzureSpeech
) -> None:
    """
    Speak text using Azure Speech API.
    Cache audio files to avoid unnecessary API calls.
    """
    text_details = speech_svc.language + speech_svc.gender + speech_svc.voice + text

    text_details_hash = hashlib.sha256(text_details.encode("utf-8")).hexdigest()

    filename = os.path.join(cache_dir, f"{text_details_hash}.wav")

    if not os.path.exists(filename):
        audio_data = speech_svc.text_to_speech(text)

        with wave.open(filename, "wb") as f:
            f.setnchannels(player.channels)

            if player.audio_format == pyaudio.paInt8:
                f.setsampwidth(1)
            elif player.audio_format == pyaudio.paInt16:
                f.setsampwidth(2)
            elif player.audio_format == pyaudio.paInt32:
                f.setsampwidth(4)
            else:
                raise ValueError("Invalid audio format")

            f.setframerate(player.rate)

            f.writeframes(audio_data)

    with wave.open(filename, "rb") as f:
        player.play(f.readframes(f.getnframes()))


def check_for_event() -> Union[str, None]:
    """
    Check for events and return a string representing the event if one is found.
    """
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                return "Quit"
            if event.key == K_SPACE:
                return "Next"
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == Button.BTN_BACK:
                return "Quit"
            if event.button == Button.BTN_A:
                return "Next"

    return None


def get_random_string(length: int) -> str:
    """
    Generate a random string of lowercase letters.

    Used for generating unique filenames.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def check_moderation(msg: str) -> bool:
    """
    Check if a message complies with content policy.

    Returns True if message is safe, False if it is not.
    """
    response = openai.Moderation.create(input=msg)

    return not response["results"][0]["flagged"]


def get_verse(system_prompt: str, base_prompt: str, user_prompt: str) -> str:
    """
    Get a verse from GPT chat completion.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt + " " + user_prompt},
    ]

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    return response["choices"][0]["message"]["content"]


def show_status_screen(
    surface: pygame.Surface, text: str, horiz_margin: int, vert_margin: int
) -> None:
    """
    Show a status screen with a message.

    TODO: Generalize positioning code and remove magic numbers
    """
    surface.fill(pygame.Color("black"))

    font = pygame.font.SysFont("Arial", 200)
    x_pos = surface.get_width() / 2 - font.size("A.R.T.I.S.T.")[0] / 2
    text_surface = font.render("A.R.T.I.S.T.", True, pygame.Color("white"))
    surface.blit(text_surface, (x_pos, vert_margin))

    font = pygame.font.SysFont("Arial", 60)
    tagline = "Audio-Responsive Transformative Imagination Synthesis Technology"
    x_pos = surface.get_width() / 2 - font.size(tagline)[0] / 2
    text_surface = font.render(tagline, True, pygame.Color("white"))
    surface.blit(text_surface, (x_pos, 250))

    font = pygame.font.SysFont("Arial", 100)
    x_pos = surface.get_width() / 2 - font.size(text)[0] / 2
    text_surface = font.render(text, True, pygame.Color("white"))
    surface.blit(text_surface, (x_pos, 500))

    pygame.display.update()


def main() -> None:
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        azure_speech_region = os.environ["AZURE_SPEECH_REGION"]
        azure_speech_key = os.environ["AZURE_SPEECH_KEY"]
    except KeyError:
        print("Please set environment variables for OpenAI and Azure Speech.")
        return

    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print("Please create a config.json file.")
        return

    cache_dir = config["speech_cache_dir"]
    transcribe_temp_dir = config["transcribe_temp_dir"]
    output_dir = config["output_dir"]

    language = config["speech_language"]
    gender = config["speech_gender"]
    voice = config["speech_voice"]
    input_sample_rate = config["input_sample_rate"]
    output_sample_rate = config["output_sample_rate"]

    max_recording_time = config["max_recording_time"]

    img_width = config["img_width"]
    img_height = config["img_height"]

    img_size = f"{img_width}x{img_height}"

    display_width = config["display_width"]
    display_height = config["display_height"]

    horiz_margin = config["horiz_margin"]
    vert_margin = config["vert_margin"]

    image_base_prompt = config["image_base_prompt"]

    verse_system_prompt = config["verse_system_prompt"]
    verse_base_prompt = config["verse_base_prompt"]

    verse_font = config["verse_font"]
    verse_font_size = config["verse_font_size"]
    verse_line_spacing = config["verse_line_spacing"]

    max_verse_width = (display_width - img_width) - (horiz_margin * 3)

    random.seed()

    disp_surface = init_display(width=display_width, height=display_height)

    init_joystick()

    speech_svc = AzureSpeech(
        subscription_key=azure_speech_key,
        region=azure_speech_region,
        language=language,
        gender=gender,
        voice=voice,
    )

    audio_player = AudioPlayer(channels=1, rate=output_sample_rate)

    audio_recorder = AudioRecorder(channels=1, rate=input_sample_rate)

    transcriber = Transcriber(
        temp_dir=transcribe_temp_dir,
        channels=1,
        sample_width=2,
        framerate=input_sample_rate,
    )

    start_new = True

    show_status_screen(disp_surface, "Ready")

    while True:
        while True:
            status = check_for_event()

            if status == "Quit":
                pygame.quit()
                return
            elif status == "Next":
                start_new = True
                break

        if start_new:
            show_status_screen(
                surface=disp_surface,
                text=" ",
                horiz_margin=horiz_margin,
                vert_margin=vert_margin,
            )
            pygame.display.update()

            greeting_phrase = (
                random.choice(config["welcome_words"])
                + " "
                + random.choice(config["welcome_lines"])
            )

            speak_text(
                text=greeting_phrase,
                cache_dir=cache_dir,
                player=audio_player,
                speech_svc=speech_svc,
            )

            show_status_screen(
                surface=disp_surface,
                text="Listening...",
                horiz_margin=horiz_margin,
                vert_margin=vert_margin,
            )

            start_new = False

        silent_loops = 0

        while silent_loops < 10:
            (in_stream, valid_audio) = audio_recorder.record(max_recording_time)

            if valid_audio:
                show_status_screen(
                    surface=disp_surface,
                    text="Working...",
                    horiz_margin=horiz_margin,
                    vert_margin=vert_margin,
                )
                working_phrase = random.choice(config["working_lines"])

                speak_text(
                    text=working_phrase,
                    cache_dir=cache_dir,
                    player=audio_player,
                    speech_svc=speech_svc,
                )

                msg = transcriber.transcribe(audio_stream=in_stream)

                name = get_random_string(12)

                with open(os.path.join(output_dir, name + ".txt"), "w") as f:
                    f.write(msg)

                img_prompt = image_base_prompt + msg

                can_create = check_moderation(img_prompt)

                if can_create:
                    response = openai.Image.create(
                        prompt=img_prompt, size=img_size, response_format="b64_json"
                    )

                    img_bytes = base64.b64decode(response["data"][0]["b64_json"])

                    verse = get_verse(
                        system_prompt=verse_system_prompt,
                        base_prompt=verse_base_prompt,
                        user_prompt=msg,
                    )

                    verse_lines = verse.split("\n")

                    verse_lines = [line.strip() for line in verse_lines]

                    longest_line = max(verse_lines, key=len)

                    font_size = verse_font_size
                    will_fit = False

                    while not will_fit:
                        font_obj = pygame.font.SysFont(verse_font, font_size)

                        text_size = font_obj.size(longest_line)

                        if text_size[0] < max_verse_width:
                            will_fit = True
                        else:
                            font_size -= 2

                    total_height = 0

                    for line in verse_lines:
                        text_size = font_obj.size(line)

                        total_height += text_size[1]
                        total_height += verse_line_spacing

                    total_height -= verse_line_spacing  # No spacing after last line

                    offset = -int(total_height / (len(verse_lines) / 2))

                    img_side = random.choice(["left", "right"])

                    disp_surface.fill(pygame.Color("black"))

                    if img_side == "left":
                        img_x = horiz_margin
                        verse_x = horiz_margin + img_width + horiz_margin
                    else:
                        img_x = display_width - horiz_margin - img_width
                        verse_x = horiz_margin

                    for line in verse_lines:
                        text_surface_obj = font_obj.render(
                            line, True, pygame.Color("white")
                        )
                        disp_surface.blit(
                            text_surface_obj, (verse_x, (display_height / 2) + offset)
                        )
                        offset += int(total_height / len(verse_lines))

                    finished_phrase = random.choice(config["finished_lines"])

                    speak_text(
                        text=finished_phrase,
                        cache_dir=cache_dir,
                        player=audio_player,
                        speech_svc=speech_svc,
                    )

                    with open(os.path.join(output_dir, name + ".png"), "wb") as f:
                        f.write(img_bytes)

                    img = pygame.image.load(os.path.join(output_dir, name + ".png"))
                    disp_surface.blit(img, (img_x, vert_margin))
                    pygame.display.update()

                    pygame.image.save(
                        disp_surface, os.path.join(output_dir, name + "-verse.png")
                    )
                else:
                    show_status_screen(
                        surface=disp_surface,
                        text="Creation failed!",
                        horiz_margin=horiz_margin,
                        vert_margin=vert_margin,
                    )
                    failed_phrase = random.choice(config["failed_lines"])

                    speak_text(
                        text=failed_phrase,
                        cache_dir=cache_dir,
                        player=audio_player,
                        speech_svc=speech_svc,
                    )

                break
            else:
                silent_loops += 1

        if silent_loops == 10:
            show_status_screen(
                surface=disp_surface,
                text="Ready",
                horiz_margin=horiz_margin,
                vert_margin=vert_margin,
            )


if __name__ == "__main__":
    main()
