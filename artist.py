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

Uses OpenAI's DALL-E 2 to generate images, GPT Chat to generate verses
and Whisper API to transcribe speech.

Uses Azure Speech API to convert text to speech.

Uses Azure Blob Storage to store downloadable images.
"""

import base64
import datetime
import hashlib
import io
import json
import logging
import random
import os
import string
import time
import wave

import pygame
import qrcode
import openai

from artist_classes import ArtistCanvas, ArtistCreation, StatusScreen
from audio_tools import AudioPlayer, AudioRecorder
from azure_speech import AzureSpeech
from azure.storage.blob import BlobServiceClient, ContentSettings
from log_config import create_global_logger
from openai_tools import ChatCharacter, Transcriber
from pygame.locals import *
from typing import Union


class ButtonConfig:
    """
    A class to hold information about game controller button mappings in order to
    avoid passing a huge number of parameters to the check_for_event function.
    """

    def __init__(
        self,
        generate_button: int,
        daydream_button: int,
        reveal_qr_button: int,
        reveal_prompt_button: int,
        shutdown_hold_button: int,
        shutdown_press_button: int,
    ) -> None:
        self.generate_button = generate_button
        self.daydream_button = daydream_button
        self.reveal_qr_button = reveal_qr_button
        self.reveal_prompt_button = reveal_prompt_button
        self.shutdown_hold_button = shutdown_hold_button
        self.shutdown_press_button = shutdown_press_button


# Global logger object to avoid passing logger to many functions
logger = create_global_logger("artist.log", logging.DEBUG)


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


def init_joystick() -> Union[pygame.joystick.JoystickType, None]:
    """
    Initialize joystick if one is connected.

    Returns joystick object if one is connected, otherwise returns None.

    The returned joystick object must remain in scope for button press events
    to be detected.
    """
    pygame.joystick.init()

    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        return joystick
    else:
        return None


def speak_text(
    text: str,
    cache_dir: str,
    player: AudioPlayer,
    speech_svc: AzureSpeech,
    use_cache: bool = True,
) -> None:
    """
    Speak text using Azure Speech API.

    Caches audio files to avoid unnecessary API calls, but allows for bypassing caching
    when speaking lines that will be unique.
    """
    logger.info(f"Speaking: {text}")
    text_details = speech_svc.language + speech_svc.gender + speech_svc.voice + text

    text_details_hash = hashlib.sha256(text_details.encode("utf-8")).hexdigest()

    logger.debug(f"Text details: {text_details} - Hash: {text_details_hash}")

    if use_cache:
        filename = os.path.join(cache_dir, f"{text_details_hash}.wav")

        if not os.path.exists(filename):
            logger.debug(f"Cache miss - generating audio file: {filename}")
            audio_data = speech_svc.text_to_speech(text)

            # Audio from Azure Speech API is already in WAV format including header
            with open(filename, "wb") as f:
                f.write(audio_data)

        with wave.open(filename, "rb") as f:
            logger.debug(f"Playing audio file: {filename}")
            player.play(f.readframes(f.getnframes()))
    else:
        audio_data = speech_svc.text_to_speech(text)

        wav_data = io.BytesIO(audio_data)

        with wave.open(wav_data, "rb") as f:
            player.play(f.readframes(f.getnframes()))


def check_for_event(
    js: Union[pygame.joystick.JoystickType, None],
    button_config: ButtonConfig,
) -> Union[str, None]:
    """
    Check for events and return a string representing the event if one is found.
    """
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                return "Quit"
            if event.key == K_SPACE:
                return "New"
            if event.key == K_d:
                return "Daydream"
            if event.key == K_p:
                return "Prompt"
            if event.key == K_q:
                return "QR"
            if event.key == K_RIGHT:
                return "Next-Recent"
            if event.key == K_LEFT:
                return "Previous-Recent"
        elif js and event.type == pygame.JOYBUTTONDOWN:
            if event.button == button_config.shutdown_press_button:
                if js.get_button(button_config.shutdown_hold_button):
                    return "Quit"
            if event.button == button_config.generate_button:
                return "New"
            if event.button == button_config.daydream_button:
                return "Daydream"
            if event.button == button_config.reveal_prompt_button:
                return "Prompt"
            if event.button == button_config.reveal_qr_button:
                return "QR"
        elif js and event.type == pygame.JOYAXISMOTION:
            if event.axis == 0 and event.value < -0.5:
                return "Previous-Recent"
            if event.axis == 0 and event.value > 0.5:
                return "Next-Recent"

    return None


def get_random_string(length: int) -> str:
    """
    Generate a random string of lowercase letters and digits.

    Used for generating unique filenames.
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def check_moderation(msg: str) -> bool:
    """
    Check if a message complies with content policy.

    Returns True if message is safe, False if it is not.
    """
    try:
        response = openai.Moderation.create(input=msg)
    except Exception as e:
        logger.error(f"Moderation response: {response}")
        logger.exception(e)
        raise

    flagged = response["results"][0]["flagged"]

    if flagged:
        logger.info(f"Message flagged by moderation: {msg}")
        logger.info(f"Moderation response: {response}")
    else:
        logger.info(f"Moderation check passed")

    return not flagged


def get_one_verse(
    poet: ChatCharacter,
    base_prompt: str,
    user_prompt: str,
) -> str:
    """
    Get one verse from poet character.
    """

    # Poet is a single-turn character so no history is needed
    poet.reset()

    try:
        verse = poet.get_chat_response(base_prompt + " " + user_prompt).content
    except Exception as e:
        logger.error(f"Error getting verse from poet")
        logger.exception(e)
        raise

    return verse


def get_best_verse(
    poet: ChatCharacter,
    critic: ChatCharacter,
    base_prompt: str,
    user_prompt: str,
    num_verses: int,
) -> str:
    """
    Get num_verse verses from poet character, then use the critic character
    to choose the best verse.
    """

    verses: list[str] = []

    for _ in range(num_verses):
        verse = get_one_verse(
            poet=poet, base_prompt=base_prompt, user_prompt=user_prompt
        )

        verses.append(verse)

    # Critic is a single-turn character so no history is needed
    critic.reset()

    critic_message = f"Theme: {user_prompt}\n"

    for verse in enumerate(verses, start=1):
        critic_message += f"Poem {verse[0]}: {verse[1]}\n"

    critic_log_message = critic_message.strip().replace("\n", "/")
    logger.info(f"Critic message: {critic_log_message}")

    chosen_poem = None

    try:
        critic_verdict = critic.get_chat_response(critic_message).content
        logger.info(f"Critic verdict: {critic_verdict}")

        for c in critic_verdict:
            if c.isdigit():
                chosen_poem = int(c)
                logger.debug(f"Chosen poem number: {chosen_poem}")
                break
    except Exception as e:
        logger.error(f"Error getting verdict from critic")
        logger.exception(e)
        raise

    if chosen_poem is not None:  # Maybe there could be an index 0 someday?
        return verses[chosen_poem - 1]
    else:
        logger.warning(
            f"No poem number found in critic verdict - returning random verse"
        )
        return random.choice(verses)


def get_prompt_surface(
    prompt: str,
    prompt_source: str,
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    margin_size: int = 10,
) -> pygame.Surface:
    """
    Get a surface with the prompt text and prompt source rendered on it.
    """
    prompt_surface = pygame.Surface((width, height))
    prompt_surface.fill(pygame.Color("yellow"))

    text_surface = pygame.Surface(
        (width - (margin_size * 2), height - (margin_size * 2))
    )
    text_surface.fill(pygame.Color("black"))

    text_subsurface = pygame.Surface(
        (width - (margin_size * 4), height - (margin_size * 4))
    )
    text_subsurface.fill(pygame.Color("black"))

    prompt = "Prompt: " + prompt
    prompt_source = "Source: " + prompt_source

    font = pygame.font.SysFont(font_name, font_size)

    prompt_words = prompt.split()

    line = ""
    y_pos = 0

    total_height = 0

    for word in prompt_words:
        previous_line = line
        line += word + " "

        line_width = font.size(line)[0]
        line_height = font.size(line)[1]

        if line_width > width - (margin_size * 8):
            line_surface = font.render(previous_line, True, pygame.Color("white"))
            logger.debug(f"Rendering word-wrapped prompt line: {previous_line}")
            text_subsurface.blit(line_surface, (margin_size, y_pos))

            line = word + " "
            y_pos += line_height
            total_height += line_height

    # Render any remaining words
    if line.strip():
        line_surface = font.render(line, True, pygame.Color("white"))
        logger.debug(f"Rendering prompt line: {line}")
        text_subsurface.blit(line_surface, (margin_size, y_pos))
        total_height += line_height

    # Leave blank line before prompt source
    y_pos += line_height * 2
    total_height += line_height

    line_surface = font.render(prompt_source, True, pygame.Color("white"))
    total_height += line_height
    logger.debug(f"Rendering prompt source line: {prompt_source}")
    text_subsurface.blit(line_surface, (margin_size, y_pos))

    text_surface.blit(
        text_subsurface,
        (margin_size, (text_subsurface.get_height() - total_height) // 2),
    )

    prompt_surface.blit(text_surface, (margin_size, margin_size))

    return prompt_surface


def show_status_screen(
    surface: pygame.Surface, text: str, status_screen_obj: StatusScreen
) -> None:
    """
    Show a status screen with a message.
    """
    status_screen_obj.render_status(text)

    update_display(surface, status_screen_obj.surface)


def update_display(
    display_surface: pygame.Surface, content_surface: pygame.Surface
) -> None:
    """
    Update the display with the content surface.
    """
    display_surface.blit(content_surface, (0, 0))
    pygame.display.update()


def load_recents(recents_file_name: str) -> list:
    try:
        with open(recents_file_name, "r") as recents_file:
            recents = json.load(recents_file)
    except FileNotFoundError:
        recents = []

    return recents


def save_recents(recents: list, recents_file_name: str) -> None:
    with open(recents_file_name, "w") as recents_file:
        json.dump(recents, recents_file, indent=4)


def main() -> None:
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        azure_speech_region = os.environ["AZURE_SPEECH_REGION"]
        azure_speech_key = os.environ["AZURE_SPEECH_KEY"]
        azure_storage_key = os.environ["AZURE_STORAGE_KEY"]
    except KeyError:
        print("Please set environment variables for OpenAI and Azure.")
        return

    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print("Please create a config.json file.")
        return

    logger.info("*** Starting A.R.T.I.S.T. ***")

    # In general, configuration items that are referenced multiple times are
    # initialized here. Items that are used only once are usually referenced directly
    # where they are used.
    cache_dir = config["speech_cache_dir"]
    output_dir = config["output_dir"]

    recents_file_name = config["recents_file_name"]

    storage_account = config["storage_account"]
    storage_container = config["storage_container"]

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

    button_config = ButtonConfig(
        generate_button=config["generate_button"],
        daydream_button=config["daydream_button"],
        reveal_qr_button=config["reveal_qr_button"],
        reveal_prompt_button=config["reveal_prompt_button"],
        shutdown_hold_button=config["shutdown_hold_button"],
        shutdown_press_button=config["shutdown_press_button"],
    )

    num_verses = config["num_verses"]

    # Critic is recommended for best results, but can be disabled to save tokens when
    # using GPT-4 chat completion API
    use_critic = config["use_critic"]

    min_daydream_time = config["min_daydream_time"] * 60  # Convert to seconds
    max_daydream_time = config["max_daydream_time"] * 60  # Convert to seconds

    manual_daydream_window = config["manual_daydream_window"] * 60  # Convert to seconds

    random.seed()

    logger.debug("Initializing display...")
    disp_surface = init_display(width=display_width, height=display_height)

    logger.debug("Initializing joystick...")
    js = init_joystick()

    logger.debug("Initializing speech...")
    speech_svc = AzureSpeech(
        subscription_key=azure_speech_key,
        region=azure_speech_region,
        language=config["speech_language"],
        gender=config["speech_gender"],
        voice=config["speech_voice"],
    )

    logger.debug("Initializing storage...")
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account}.blob.core.windows.net",
        credential=azure_storage_key,
    )
    blob_container_client = blob_service_client.get_container_client(
        container=storage_container
    )

    logger.debug("Initializing audio player...")
    audio_player = AudioPlayer(sample_width=2, channels=1, rate=output_sample_rate)

    logger.debug("Initializing audio recorder...")
    audio_recorder = AudioRecorder(sample_width=2, channels=1, rate=input_sample_rate)

    logger.debug("Initializing transcriber...")
    transcriber = Transcriber(
        channels=1,
        sample_width=2,
        framerate=input_sample_rate,
        model=config["transcriber_model"],
    )

    logger.debug("Initialzing autonomous AI artist...")
    ai_artist = ChatCharacter(
        system_prompt=config["artist_system_prompt"], model=config["artist_chat_model"]
    )

    logger.debug("Initializing poet...")
    poet = ChatCharacter(
        system_prompt=config["poet_system_prompt"], model=config["poet_chat_model"]
    )

    if use_critic:
        logger.debug("Initializing critic...")
        critic = ChatCharacter(
            system_prompt=config["critic_system_prompt"],
            model=config["critic_chat_model"],
        )

    logger.debug("Initializing artist canvas...")
    artist_canvas = ArtistCanvas(
        width=display_width,
        height=display_height,
        horiz_margin=horiz_margin,
        vert_margin=vert_margin,
        verse_font_name=config["verse_font"],
        verse_font_max_size=config["verse_font_size"],
        verse_line_spacing=config["verse_line_spacing"],
    )

    logger.debug("Initializing status screen...")
    status_screen = StatusScreen(
        width=display_width,
        height=display_height,
        font_name=config["status_font"],
        heading1_size=config["status_heading1_size"],
        heading2_size=config["status_heading2_size"],
        status_size=config["status_status_size"],
        vert_margin=vert_margin,
    )

    logger.debug("Loading recent creations...")
    recents = load_recents(recents_file_name)
    recent_index = 0

    daydream = False
    manual_daydream_timestamps = []

    previous_user_prompt = ""
    user_prompt = ""

    base_file_name = None

    show_status_screen(
        surface=disp_surface, text="Ready", status_screen_obj=status_screen
    )

    next_change_time = time.monotonic() + random.randint(
        min_daydream_time, max_daydream_time
    )

    while True:
        # Clear any accumulated events
        _ = pygame.event.get()

        while True:
            # Slow down loop to reduce power consumption
            time.sleep(0.1)

            if manual_daydream_timestamps:
                if (
                    time.monotonic() - manual_daydream_timestamps[0]
                    > manual_daydream_window
                ):
                    logger.debug(
                        f"Removing expired daydream timestamp {manual_daydream_timestamps[0]} at {time.monotonic()}"
                    )
                    manual_daydream_timestamps = manual_daydream_timestamps[1:]

            status = check_for_event(
                js=js,
                button_config=button_config,
            )

            time_now = datetime.datetime.now()

            if (
                time_now.hour >= config["daydream_start_hour"]
                and time_now.hour < config["daydream_end_hour"]
                and time_now.isoweekday() in config["daydream_iso_weekdays"]
                and time.monotonic() >= next_change_time
            ):
                status = "Auto-Daydream"
                daydream = True
                break

            if status == "Quit":
                logger.info("*** A.R.T.I.S.T. is shutting down. ***")

                # This is a workaround for crash-to-desktop issues until the code
                # can be refactored for better error handling. A shell script should
                # check for this file, and if it does not exist, restart the program.
                with open("exit-requested.txt", "w") as f:
                    f.write(
                        "This file is used to signal that the user has requested A.R.T.I.S.T. to shut down.\n"
                    )

                pygame.quit()
                return
            elif status == "New":
                daydream = False
                break
            elif status == "Daydream":
                if len(manual_daydream_timestamps) < config["manual_daydream_limit"]:
                    daydream_timestamp = time.monotonic()

                    logger.debug(f"Manual daydream request at {daydream_timestamp}.")
                    manual_daydream_timestamps.append(daydream_timestamp)
                    daydream = True
                    break
                else:
                    speak_text(
                        text=random.choice(config["daydream_refusal_lines"]),
                        cache_dir=cache_dir,
                        player=audio_player,
                        speech_svc=speech_svc,
                    )
                    logger.debug("Manual daydream request refused.")
            elif status == "Prompt":
                if base_file_name:
                    prompt_surface = get_prompt_surface(
                        prompt=user_prompt,
                        prompt_source="User prompt"
                        if not daydream
                        else "A.R.T.I.S.T. Daydream",
                        width=int(display_width * 0.75),
                        height=int(display_height * 0.4),
                        font_name=config["prompt_font"],
                        font_size=config["prompt_font_size"],
                    )

                    x_pos = int((display_width - prompt_surface.get_width()) / 2)
                    y_pos = int((display_height - prompt_surface.get_height()) / 2)

                    disp_surface.blit(prompt_surface, (x_pos, y_pos))
                    pygame.display.update()

                    time.sleep(config["prompt_display_time"])

                    disp_surface.blit(artist_canvas.surface, (0, 0))
                    pygame.display.update()
            elif status == "QR":
                # Quick way to make sure a creation has already been generated
                if base_file_name:
                    img_url = f"https://{storage_account}.blob.core.windows.net/{storage_container}/{base_file_name}.html"
                    qr_img = qrcode.make(img_url)
                    qr_img_data = io.BytesIO()

                    qr_img.save(qr_img_data, format="PNG")

                    # Need to return pointer to start of data before reading
                    qr_img_data.seek(0)

                    # "qr.png" is a name hint to assist in file format detection, not
                    # an actual file on disk
                    qr_surf = pygame.image.load(qr_img_data, "qr.png")

                    qr_width = qr_surf.get_width()
                    qr_height = qr_surf.get_height()

                    qr_x_pos = (display_width - qr_width) // 2
                    qr_y_pos = (display_height - qr_height) // 2

                    disp_surface.blit(qr_surf, (qr_x_pos, qr_y_pos))
                    pygame.display.update()
                    time.sleep(config["qr_display_time"])

                    disp_surface.blit(artist_canvas.surface, (0, 0))
                    pygame.display.update()

                    # Don't break out of the loop after QR has been shown since no
                    # further action is required
            elif status in ["Previous-Recent", "Next-Recent"]:
                if recents:
                    if status == "Previous-Recent":
                        recent_index = (recent_index - 1) % len(recents)
                    else:
                        recent_index = (recent_index + 1) % len(recents)

                    base_file_name = recents[recent_index]["base_name"]

                    user_prompt = recents[recent_index]["prompt"]
                    previous_user_prompt = user_prompt

                    daydream = recents[recent_index]["daydream"]

                    recent_img = pygame.image.load(
                        os.path.join(output_dir, f"{base_file_name}.png")
                    )

                    artist_canvas.surface.blit(recent_img, (0, 0))
                    disp_surface.blit(artist_canvas.surface, (0, 0))
                    pygame.display.update()

        if not daydream:
            logger.info("=== Starting new creation ===")

            show_status_screen(
                surface=disp_surface, text=" ", status_screen_obj=status_screen
            )

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
                status_screen_obj=status_screen,
            )

            logger.debug("Recording...")

            silent_loops = 0
            audio_detected = False

            while silent_loops < 10:
                (in_stream, valid_audio) = audio_recorder.record(max_recording_time)

                if valid_audio:
                    audio_detected = True
                    show_status_screen(
                        surface=disp_surface,
                        text="Working...",
                        status_screen_obj=status_screen,
                    )
                    working_phrase = random.choice(config["working_lines"])

                    speak_text(
                        text=working_phrase,
                        cache_dir=cache_dir,
                        player=audio_player,
                        speech_svc=speech_svc,
                    )

                    user_prompt = transcriber.transcribe(audio_stream=in_stream)

                    logger.info(f"Transcribed: {user_prompt}")

                    break
                else:
                    silent_loops += 1

            if not audio_detected:
                logger.debug("Silence detected")
                show_status_screen(
                    surface=disp_surface,
                    text="Ready",
                    status_screen_obj=status_screen,
                )
                continue
        else:
            logger.info("=== Starting daydream ===")
            ai_artist.reset()

            show_status_screen(
                surface=disp_surface,
                text="Daydreaming...",
                status_screen_obj=status_screen,
            )

            # Only speak line if daydream is manually initiated
            if status == "Daydream":
                speak_text(
                    text=random.choice(config["daydream_lines"]),
                    cache_dir=cache_dir,
                    player=audio_player,
                    speech_svc=speech_svc,
                )

            if previous_user_prompt:
                daydream_prompt = previous_user_prompt
            else:
                daydream_prompt = " something completely random."

            logger.debug(f"Daydreaming based on: {daydream_prompt}")
            user_prompt = ai_artist.get_chat_response(
                message=config["artist_base_prompt"] + " " + daydream_prompt
            ).content

            logger.info(f"Daydreamed: {user_prompt}")

        base_file_name = get_random_string(config["file_name_length"])

        logger.info(f"Base name: {base_file_name}")

        img_prompt = random.choice(config["image_base_prompts"]) + user_prompt
        previous_user_prompt = user_prompt

        can_create = check_moderation(img_prompt)
        creation_failed = False
        response = None  # Clear out previous response

        if can_create:
            try:
                response = openai.Image.create(
                    prompt=img_prompt,
                    size=img_size,
                    response_format="b64_json",
                    user="A.R.T.I.S.T.",
                )
            except Exception as e:
                logger.error(f"Image creation response: {response}")
                logger.exception(e)
                creation_failed = True

            if not creation_failed:
                img_bytes = base64.b64decode(response["data"][0]["b64_json"])

                if use_critic:
                    logger.debug("Getting best verse...")
                    verse = get_best_verse(
                        poet=poet,
                        critic=critic,
                        base_prompt=config["verse_base_prompt"],
                        user_prompt=user_prompt,
                        num_verses=num_verses,
                    )
                else:
                    logger.debug("Getting one verse...")
                    verse = get_one_verse(
                        poet=poet,
                        base_prompt=config["verse_base_prompt"],
                        user_prompt=user_prompt,
                    )

                verse_lines = verse.split("\n")

                verse_lines = [line.strip() for line in verse_lines]
                logger.info(f"Verse: {'/'.join(verse_lines)}")

                img_side = random.choice(["left", "right"])

                finished_phrase = random.choice(config["finished_lines"])

                if not daydream:
                    speak_text(
                        text=finished_phrase,
                        cache_dir=cache_dir,
                        player=audio_player,
                        speech_svc=speech_svc,
                    )

                # Only speak prompt if daydream was manually initiated
                elif status == "Daydream":
                    speak_text(
                        text=user_prompt,
                        cache_dir=cache_dir,
                        player=audio_player,
                        speech_svc=speech_svc,
                        use_cache=False,
                    )

                logger.debug("Saving image...")
                raw_image_file_name = base_file_name + "-raw.png"

                with open(os.path.join(output_dir, raw_image_file_name), "wb") as f:
                    f.write(img_bytes)

                img = pygame.image.load(os.path.join(output_dir, raw_image_file_name))

                creation = ArtistCreation(img, verse_lines, user_prompt, daydream)
                artist_canvas.render_creation(creation, img_side)
                disp_surface.blit(artist_canvas.surface, (0, 0))
                pygame.display.update()

                logger.debug("Saving creation...")
                screenshot_file_name = base_file_name + ".png"
                pygame.image.save(
                    disp_surface, os.path.join(output_dir, screenshot_file_name)
                )

                logger.debug("Uploading creation...")
                image_url = f"https://{storage_account}.blob.core.windows.net/{storage_container}/{screenshot_file_name}"

                html_file_name = base_file_name + ".html"

                with open(config["html_template"], "r") as template_file:
                    with open(
                        os.path.join(output_dir, html_file_name), "w"
                    ) as output_html_file:
                        for line in template_file:
                            out_line = line.replace("***IMG-URL***", image_url)
                            out_line = out_line.replace("***PROMPT***", user_prompt)
                            out_line = out_line.replace(
                                "***GEN-BY***",
                                "A.R.T.I.S.T. Daydream" if daydream else "User Request",
                            )
                            out_line = out_line.replace("***TIME***", time.asctime())

                            output_html_file.write(out_line)

                with open(os.path.join(output_dir, html_file_name), "rb") as f:
                    try:
                        content_settings = ContentSettings(content_type="text/html")
                        blob_container_client.upload_blob(
                            name=base_file_name + ".html",
                            data=f,
                            overwrite=True,
                            content_settings=content_settings,
                        )
                    except Exception as e:
                        logger.error("Error uploading HTML to blob storage")
                        logger.exception(e)

                with open(os.path.join(output_dir, screenshot_file_name), "rb") as f:
                    try:
                        content_settings = ContentSettings(content_type="image/png")
                        blob_container_client.upload_blob(
                            name=screenshot_file_name,
                            data=f,
                            overwrite=True,
                            content_settings=content_settings,
                        )
                    except Exception as e:
                        logger.error("Error uploading screenshot to blob storage")
                        logger.exception(e)

                logger.debug("Updating recent creations...")
                recents.append(
                    {
                        "base_name": base_file_name,
                        "prompt": user_prompt,
                        "daydream": daydream,
                    }
                )

                if len(recents) > config["max_recents"]:
                    recents = recents[-config["max_recents"] :]

                save_recents(recents, recents_file_name)

                recent_index = len(recents) - 1

                next_change_time = time.monotonic() + random.randint(
                    min_daydream_time, max_daydream_time
                )

        if not can_create or creation_failed:
            show_status_screen(
                surface=disp_surface,
                text="Creation failed!",
                status_screen_obj=status_screen,
            )
            failed_phrase = random.choice(config["failed_lines"])

            speak_text(
                text=failed_phrase,
                cache_dir=cache_dir,
                player=audio_player,
                speech_svc=speech_svc,
            )


if __name__ == "__main__":
    main()
