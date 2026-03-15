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

Uses OpenAI DALL-E 2/DALL-E 3 or Stability AI SDXL/Stable Image to generate images.

Uses OpenAI GPT Chat Completion or Anthropic Claude to generate verses and
daydream image prompts, and to choose the best verse from multiple options if
the critic is enabled.

Uses Whisper API to transcribe speech.

Uses Azure Speech API to convert text to speech.

Uses Azure Blob Storage to store downloadable images.
"""

import argparse
import datetime
import io
import json
import logging
import os
import random
import string
import time
from dataclasses import dataclass, field
from enum import Enum

import pygame
import qrcode

from pygame.locals import *

from artist_classes import (
    ArtistCanvas,
    ArtistCreation,
    DallE2Creator,
    DallE3Creator,
    GptImage1Creator,
    SDXLCreator,
    StableImageCreator,
    StatusScreen,
)
from artist_moderator import ArtistModerator
from artist_speech import ArtistSpeech
from artist_storage import ArtistStorage
from audio_tools import AudioRecorder
from log_config import create_global_logger
from openai_tools import ChatCharacter, Transcriber
from anthropic_tools import ClaudeChatCharacter


# Global logger object to avoid passing logger to many functions
logger = create_global_logger("artist.log", logging.DEBUG)


@dataclass
class ButtonConfig:
    """
    Game controller button mappings for the check_for_event function.
    """

    generate_button: int
    daydream_button: int
    reveal_qr_button: int
    reveal_prompt_button: int
    shutdown_hold_button: int
    shutdown_press_button: int


class UserAction(Enum):
    """
    An enumeration of user actions.
    """

    QUIT = 1
    NEW = 2
    DAYDREAM = 3
    SHOW_PROMPT = 4
    SHOW_QR = 5
    PREVIOUS_RECENT = 6
    NEXT_RECENT = 7
    AUTO_DAYDREAM = 8


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


def init_joystick() -> pygame.joystick.JoystickType | None:
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


def check_for_event(
    js: pygame.joystick.JoystickType | None,
    button_config: ButtonConfig,
) -> UserAction | None:
    """
    Check for events and return a string representing the event if one is found.
    """
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                return UserAction.QUIT
            if event.key == K_SPACE:
                return UserAction.NEW
            if event.key == K_d:
                return UserAction.DAYDREAM
            if event.key == K_p:
                return UserAction.SHOW_PROMPT
            if event.key == K_q:
                return UserAction.SHOW_QR
            if event.key == K_RIGHT:
                return UserAction.NEXT_RECENT
            if event.key == K_LEFT:
                return UserAction.PREVIOUS_RECENT
        elif js and event.type == pygame.JOYBUTTONDOWN:
            if event.button == button_config.shutdown_press_button:
                if js.get_button(button_config.shutdown_hold_button):
                    return UserAction.QUIT
            if event.button == button_config.generate_button:
                return UserAction.NEW
            if event.button == button_config.daydream_button:
                return UserAction.DAYDREAM
            if event.button == button_config.reveal_prompt_button:
                return UserAction.SHOW_PROMPT
            if event.button == button_config.reveal_qr_button:
                return UserAction.SHOW_QR
        elif js and event.type == pygame.JOYAXISMOTION:
            if event.axis == 0 and event.value < -0.5:
                return UserAction.PREVIOUS_RECENT
            if event.axis == 0 and event.value > 0.5:
                return UserAction.NEXT_RECENT

    return None


def get_random_string(length: int) -> str:
    """
    Generate a random string of lowercase letters and digits.

    Used for generating unique filenames.
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


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
    """
    Load recent creations from JSON file if the file exists,
    otherwise return an empty list.
    """
    try:
        with open(recents_file_name, "r") as recents_file:
            recents = json.load(recents_file)
    except FileNotFoundError:
        recents = []

    return recents


def save_recents(recents: list, recents_file_name: str) -> None:
    with open(recents_file_name, "w") as recents_file:
        json.dump(recents, recents_file, indent=4)


@dataclass
class AppConfig:
    """
    All configuration values loaded from the config file and environment variables.
    """

    # AI service
    chat_service: str

    # LLM - artist / daydream
    artist_chat_model: str
    artist_system_prompt: str
    artist_base_prompt: str

    # LLM - poet
    poet_chat_model: str
    poet_system_prompt: str
    verse_base_prompt: str

    # Image generation
    image_model: str
    daydream_image_model: str
    img_width: int
    img_height: int
    image_base_prompts: list

    # Display
    display_width: int
    display_height: int
    horiz_margin: int
    vert_margin: int
    verse_font: str
    verse_font_size: int
    verse_line_spacing: int
    status_font: str
    status_heading1_size: int
    status_heading2_size: int
    status_status_size: int
    prompt_font: str
    prompt_font_size: int
    prompt_display_time: float
    qr_display_time: float

    # Audio and speech
    input_sample_rate: int
    max_recording_time: int
    transcriber_model: str
    speech_language: str
    speech_gender: str
    speech_voice: str
    speech_cache_dir: str

    # Storage
    output_dir: str
    storage_account: str
    storage_container: str
    html_template: str
    file_name_length: int

    # Recents
    recents_file_name: str
    max_recents: int
    num_recents_for_daydream: int

    # Daydream scheduling (times stored in seconds)
    min_daydream_time: int
    max_daydream_time: int
    daydream_start_hour: int
    daydream_end_hour: int
    daydream_iso_weekdays: list
    manual_daydream_window: int
    manual_daydream_limit: int

    # Controller buttons
    generate_button: int
    daydream_button: int
    reveal_qr_button: int
    reveal_prompt_button: int
    shutdown_hold_button: int
    shutdown_press_button: int

    # UI text / response lines
    welcome_words: list
    welcome_lines: list
    daydream_lines: list
    working_lines: list
    finished_lines: list
    failed_lines: list
    daydream_refusal_lines: list

    # API keys (from environment variables)
    openai_api_key: str
    azure_speech_region: str
    azure_speech_key: str
    azure_storage_key: str

    # Optional fields with defaults
    poet_temperature: float = 1.0
    poet_presence_penalty: float = 0.0
    poet_frequency_penalty: float = 0.0
    num_verses: int = 3
    use_critic: bool = False
    critic_chat_model: str | None = None
    critic_system_prompt: str | None = None
    use_poem_as_user_prompt: bool = False
    use_poem_as_daydream_prompt: bool = False
    enhancer_chat_model: str | None = None
    user_prompt_enhancement_type: str | None = None
    daydream_prompt_enhancement_type: str | None = None
    enhancer_system_prompt: str | None = None
    llm_enhancer_base_prompt: str | None = None
    token_enhancer_base_prompt: str | None = None
    sdxl_steps: int | None = None
    sdxl_cfg_scale: float | None = None
    dalle3_quality: str | None = None
    gptimage1_quality: str | None = None
    stable_image_svc: str | None = None
    sd3_model: str | None = None
    use_stable_core_presets: bool = False
    stable_core_style_presets: list = field(default_factory=list)
    anthropic_api_key: str | None = None
    stability_ai_api_key: str | None = None


@dataclass
class AppState:
    """
    Mutable runtime state for the main application loop.
    """

    daydream: bool = False
    manual_daydream_timestamps: list = field(default_factory=list)
    base_file_name: str | None = None
    user_prompt: str = ""
    previous_user_prompt: str = ""
    recents: list = field(default_factory=list)
    recent_index: int = 0
    next_change_time: float = 0.0


def load_config(path: str) -> AppConfig | None:
    """
    Load, validate, and return configuration from a JSON file and environment variables.
    """
    try:
        with open(path, "r") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        return None

    required_keys = [
        "image_model",
        "daydream_image_model",
        "chat_service",
        "output_dir",
        "recents_file_name",
        "file_name_length",
        "storage_account",
        "storage_container",
        "html_template",
        "input_sample_rate",
        "max_recording_time",
        "transcriber_model",
        "img_width",
        "img_height",
        "display_width",
        "display_height",
        "horiz_margin",
        "vert_margin",
        "verse_font",
        "verse_font_size",
        "verse_line_spacing",
        "status_font",
        "status_heading1_size",
        "status_heading2_size",
        "status_status_size",
        "prompt_font",
        "prompt_font_size",
        "qr_display_time",
        "prompt_display_time",
        "speech_language",
        "speech_gender",
        "speech_voice",
        "speech_cache_dir",
        "artist_chat_model",
        "artist_system_prompt",
        "artist_base_prompt",
        "poet_chat_model",
        "poet_system_prompt",
        "verse_base_prompt",
        "min_daydream_time",
        "max_daydream_time",
        "daydream_start_hour",
        "daydream_end_hour",
        "daydream_iso_weekdays",
        "manual_daydream_window",
        "manual_daydream_limit",
        "max_recents",
        "num_recents_for_daydream",
        "generate_button",
        "daydream_button",
        "reveal_qr_button",
        "reveal_prompt_button",
        "shutdown_hold_button",
        "shutdown_press_button",
        "image_base_prompts",
        "welcome_words",
        "welcome_lines",
        "daydream_lines",
        "working_lines",
        "finished_lines",
        "failed_lines",
        "daydream_refusal_lines",
    ]

    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        print(f"Missing required config keys: {', '.join(missing_keys)}")
        return None

    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        print("Please set OPENAI_API_KEY environment variable for OpenAI API key.")
        return None

    try:
        azure_speech_region = os.environ["AZURE_SPEECH_REGION"]
        azure_speech_key = os.environ["AZURE_SPEECH_KEY"]
        azure_storage_key = os.environ["AZURE_STORAGE_KEY"]
    except KeyError:
        print(
            "Please set environment variables for Azure API keys: AZURE_SPEECH_REGION, AZURE_SPEECH_KEY, AZURE_STORAGE_KEY."
        )
        return None

    anthropic_api_key = None
    if config["chat_service"] == "anthropic":
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
            "CLAUDE_API_KEY"
        )
        if anthropic_api_key is None:
            print(
                "Please set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable for Anthropic API key."
            )
            return None

    stability_ai_api_key = None
    if config["image_model"] in ["sdxl", "stableimage"] or config[
        "daydream_image_model"
    ] in ["sdxl", "stableimage"]:
        try:
            stability_ai_api_key = os.environ["SAI_API_KEY"]
        except KeyError:
            print(
                "Please set SAI_API_KEY environment variable for Stability AI API key."
            )
            return None

    if config["img_width"] != config["img_height"]:
        print(
            "Currently only square images are supported (img_width must equal img_height)."
        )
        logger.error("img_width must equal img_height.")
        return None

    return AppConfig(
        chat_service=config["chat_service"],
        artist_chat_model=config["artist_chat_model"],
        artist_system_prompt=config["artist_system_prompt"],
        artist_base_prompt=config["artist_base_prompt"],
        poet_chat_model=config["poet_chat_model"],
        poet_system_prompt=config["poet_system_prompt"],
        verse_base_prompt=config["verse_base_prompt"],
        poet_temperature=config.get("poet_temperature", 1.0),
        poet_presence_penalty=config.get("poet_presence_penalty", 0.0),
        poet_frequency_penalty=config.get("poet_frequency_penalty", 0.0),
        num_verses=config.get("num_verses", 3),
        use_critic=config.get("use_critic", False),
        critic_chat_model=config.get("critic_chat_model"),
        critic_system_prompt=config.get("critic_system_prompt"),
        image_model=config["image_model"],
        daydream_image_model=config["daydream_image_model"],
        img_width=config["img_width"],
        img_height=config["img_height"],
        image_base_prompts=config["image_base_prompts"],
        use_poem_as_user_prompt=config.get("use_poem_as_user_prompt", False),
        use_poem_as_daydream_prompt=config.get("use_poem_as_daydream_prompt", False),
        enhancer_chat_model=config.get("enhancer_chat_model"),
        user_prompt_enhancement_type=config.get("user_prompt_enhancement_type"),
        daydream_prompt_enhancement_type=config.get("daydream_prompt_enhancement_type"),
        enhancer_system_prompt=config.get("enhancer_system_prompt"),
        llm_enhancer_base_prompt=config.get("llm_enhancer_base_prompt"),
        token_enhancer_base_prompt=config.get("token_enhancer_base_prompt"),
        sdxl_steps=config.get("sdxl_steps"),
        sdxl_cfg_scale=config.get("sdxl_cfg_scale"),
        dalle3_quality=config.get("dalle3_quality"),
        gptimage1_quality=config.get("gptimage1_quality"),
        stable_image_svc=config.get("stableimage_svc"),
        sd3_model=config.get("sd3_model"),
        use_stable_core_presets=config.get("use_stable_core_presets", False),
        stable_core_style_presets=config.get("stable_core_style_presets", []),
        display_width=config["display_width"],
        display_height=config["display_height"],
        horiz_margin=config["horiz_margin"],
        vert_margin=config["vert_margin"],
        verse_font=config["verse_font"],
        verse_font_size=config["verse_font_size"],
        verse_line_spacing=config["verse_line_spacing"],
        status_font=config["status_font"],
        status_heading1_size=config["status_heading1_size"],
        status_heading2_size=config["status_heading2_size"],
        status_status_size=config["status_status_size"],
        prompt_font=config["prompt_font"],
        prompt_font_size=config["prompt_font_size"],
        prompt_display_time=config["prompt_display_time"],
        qr_display_time=config["qr_display_time"],
        input_sample_rate=config["input_sample_rate"],
        max_recording_time=config["max_recording_time"],
        transcriber_model=config["transcriber_model"],
        speech_language=config["speech_language"],
        speech_gender=config["speech_gender"],
        speech_voice=config["speech_voice"],
        speech_cache_dir=config["speech_cache_dir"],
        output_dir=config["output_dir"],
        storage_account=config["storage_account"],
        storage_container=config["storage_container"],
        html_template=config["html_template"],
        file_name_length=config["file_name_length"],
        recents_file_name=config["recents_file_name"],
        max_recents=config["max_recents"],
        num_recents_for_daydream=config["num_recents_for_daydream"],
        min_daydream_time=config["min_daydream_time"] * 60,
        max_daydream_time=config["max_daydream_time"] * 60,
        daydream_start_hour=config["daydream_start_hour"],
        daydream_end_hour=config["daydream_end_hour"],
        daydream_iso_weekdays=config["daydream_iso_weekdays"],
        manual_daydream_window=config["manual_daydream_window"] * 60,
        manual_daydream_limit=config["manual_daydream_limit"],
        generate_button=config["generate_button"],
        daydream_button=config["daydream_button"],
        reveal_qr_button=config["reveal_qr_button"],
        reveal_prompt_button=config["reveal_prompt_button"],
        shutdown_hold_button=config["shutdown_hold_button"],
        shutdown_press_button=config["shutdown_press_button"],
        welcome_words=config["welcome_words"],
        welcome_lines=config["welcome_lines"],
        daydream_lines=config["daydream_lines"],
        working_lines=config["working_lines"],
        finished_lines=config["finished_lines"],
        failed_lines=config["failed_lines"],
        daydream_refusal_lines=config["daydream_refusal_lines"],
        openai_api_key=openai_api_key,
        azure_speech_region=azure_speech_region,
        azure_speech_key=azure_speech_key,
        azure_storage_key=azure_storage_key,
        anthropic_api_key=anthropic_api_key,
        stability_ai_api_key=stability_ai_api_key,
    )


def wait_for_action(
    cfg: AppConfig,
    js,
    button_config: ButtonConfig,
    speech_svc: ArtistSpeech,
    disp_surface: pygame.Surface,
    artist_canvas: ArtistCanvas,
    state: AppState,
) -> UserAction | None:
    """
    Block until a user action or auto-daydream trigger.

    Handles display updates for SHOW_PROMPT, SHOW_QR, and recents navigation
    as side effects. Mutates state in place. Returns the UserAction that
    triggered exit, or UserAction.QUIT if the user requested shutdown.
    """
    while True:
        time.sleep(0.1)

        if state.manual_daydream_timestamps:
            if (
                time.monotonic() - state.manual_daydream_timestamps[0]
                > cfg.manual_daydream_window
            ):
                logger.debug(
                    f"Removing expired daydream timestamp {state.manual_daydream_timestamps[0]} at {time.monotonic()}"
                )
                state.manual_daydream_timestamps = state.manual_daydream_timestamps[1:]

        user_action = check_for_event(js=js, button_config=button_config)

        time_now = datetime.datetime.now()

        if (
            time_now.hour >= cfg.daydream_start_hour
            and time_now.hour < cfg.daydream_end_hour
            and time_now.isoweekday() in cfg.daydream_iso_weekdays
            and time.monotonic() >= state.next_change_time
        ):
            state.daydream = True
            return UserAction.AUTO_DAYDREAM

        if user_action == UserAction.QUIT:
            return UserAction.QUIT
        elif user_action == UserAction.NEW:
            state.daydream = False
            return UserAction.NEW
        elif user_action == UserAction.DAYDREAM:
            if len(state.manual_daydream_timestamps) < cfg.manual_daydream_limit:
                daydream_timestamp = time.monotonic()
                logger.debug(f"Manual daydream request at {daydream_timestamp}.")
                state.manual_daydream_timestamps.append(daydream_timestamp)
                state.daydream = True
                return UserAction.DAYDREAM
            else:
                speech_svc.speak_text(text=random.choice(cfg.daydream_refusal_lines))
                logger.debug("Manual daydream request refused.")
        elif user_action == UserAction.SHOW_PROMPT:
            if state.base_file_name:
                prompt_surface = get_prompt_surface(
                    prompt=state.user_prompt,
                    prompt_source=(
                        "User prompt" if not state.daydream else "A.R.T.I.S.T. Daydream"
                    ),
                    width=int(cfg.display_width * 0.75),
                    height=int(cfg.display_height * 0.4),
                    font_name=cfg.prompt_font,
                    font_size=cfg.prompt_font_size,
                )

                x_pos = int((cfg.display_width - prompt_surface.get_width()) / 2)
                y_pos = int((cfg.display_height - prompt_surface.get_height()) / 2)

                disp_surface.blit(prompt_surface, (x_pos, y_pos))
                pygame.display.update()
                time.sleep(cfg.prompt_display_time)
                disp_surface.blit(artist_canvas.surface, (0, 0))
                pygame.display.update()
        elif user_action == UserAction.SHOW_QR:
            # Quick way to make sure a creation has already been generated
            if state.base_file_name:
                img_url = f"https://{cfg.storage_account}.blob.core.windows.net/{cfg.storage_container}/{state.base_file_name}.html"
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
                qr_x_pos = (cfg.display_width - qr_width) // 2
                qr_y_pos = (cfg.display_height - qr_height) // 2

                disp_surface.blit(qr_surf, (qr_x_pos, qr_y_pos))
                pygame.display.update()
                time.sleep(cfg.qr_display_time)
                disp_surface.blit(artist_canvas.surface, (0, 0))
                pygame.display.update()

                # Don't return after QR has been shown since no further action is required
        elif user_action in [UserAction.PREVIOUS_RECENT, UserAction.NEXT_RECENT]:
            if state.recents:
                if user_action == UserAction.PREVIOUS_RECENT:
                    state.recent_index = (state.recent_index - 1) % len(state.recents)
                else:
                    state.recent_index = (state.recent_index + 1) % len(state.recents)

                state.base_file_name = state.recents[state.recent_index]["base_name"]
                state.user_prompt = state.recents[state.recent_index]["prompt"]
                state.previous_user_prompt = state.user_prompt
                state.daydream = state.recents[state.recent_index]["daydream"]

                recent_img = pygame.image.load(
                    os.path.join(cfg.output_dir, f"{state.base_file_name}.png")
                )
                artist_canvas.surface.blit(recent_img, (0, 0))
                disp_surface.blit(artist_canvas.surface, (0, 0))
                pygame.display.update()


def capture_user_audio(
    cfg: AppConfig,
    speech_svc: ArtistSpeech,
    audio_recorder: AudioRecorder,
    transcriber: Transcriber,
    disp_surface: pygame.Surface,
    status_screen: StatusScreen,
    state: AppState,
) -> bool:
    """
    Record and transcribe user audio. Mutates state.user_prompt. Returns True if silence detected.
    """
    show_status_screen(surface=disp_surface, text=" ", status_screen_obj=status_screen)
    greeting_phrase = (
        random.choice(cfg.welcome_words) + " " + random.choice(cfg.welcome_lines)
    )
    speech_svc.speak_text(text=greeting_phrase)
    show_status_screen(
        surface=disp_surface, text="Listening...", status_screen_obj=status_screen
    )
    logger.debug("Recording...")

    silent_loops = 0
    audio_detected = False

    while silent_loops < 10:
        (in_stream, valid_audio) = audio_recorder.record(cfg.max_recording_time)
        if valid_audio:
            audio_detected = True
            show_status_screen(
                surface=disp_surface, text="Working...", status_screen_obj=status_screen
            )
            speech_svc.speak_text(text=random.choice(cfg.working_lines))
            state.user_prompt = transcriber.transcribe(audio_stream=in_stream)
            logger.info(f"Transcribed: {state.user_prompt}")
            break
        else:
            silent_loops += 1

    if not audio_detected:
        logger.debug("Silence detected")
        show_status_screen(
            surface=disp_surface, text="Ready", status_screen_obj=status_screen
        )
        return True

    return False


def generate_daydream_prompt(
    cfg: AppConfig,
    state: AppState,
    ai_artist,
    speech_svc: ArtistSpeech,
    user_action: UserAction | None,
    disp_surface: pygame.Surface,
    status_screen: StatusScreen,
) -> None:
    """
    Generate an AI prompt from recent creations, storing the result in state.user_prompt.
    """
    ai_artist.reset()
    show_status_screen(
        surface=disp_surface, text="Daydreaming...", status_screen_obj=status_screen
    )

    # Only speak line if daydream is manually initiated
    if user_action == UserAction.DAYDREAM:
        speech_svc.speak_text(text=random.choice(cfg.daydream_lines))

    if len(state.recents) >= cfg.num_recents_for_daydream:
        daydream_prompt = " , ".join(
            [r["prompt"] for r in state.recents[-cfg.num_recents_for_daydream :]]
        )
    elif len(state.recents) > 0:
        daydream_prompt = " , ".join([r["prompt"] for r in state.recents])
    else:
        daydream_prompt = " something completely random."

    logger.debug(f"Daydreaming based on: {daydream_prompt}")
    state.user_prompt = ai_artist.get_chat_response(
        message=cfg.artist_base_prompt + " " + daydream_prompt
    ).content
    logger.info(f"Daydreamed: {state.user_prompt}")


def generate_verse(cfg: AppConfig, poet, critic, state: AppState) -> str:
    """
    Generate a verse using the poet, optionally with critic selection.
    """
    if cfg.use_critic:
        logger.debug("Getting best verse...")
        return get_best_verse(
            poet=poet,
            critic=critic,
            base_prompt=cfg.verse_base_prompt,
            user_prompt=state.user_prompt,
            num_verses=cfg.num_verses,
        )
    else:
        logger.debug("Getting one verse...")
        return get_one_verse(
            poet=poet,
            base_prompt=cfg.verse_base_prompt,
            user_prompt=state.user_prompt,
        )


def enhance_poem(enhancer, base_prompt: str | None, verse: str) -> str:
    """
    Enhance a poem using the enhancer character to produce an image generation prompt.
    Falls back to the original verse if enhancement fails.
    """
    enhancer.reset()

    try:
        return enhancer.get_chat_response((base_prompt or "") + verse).content
    except Exception as e:
        logger.error("Error enhancing poem")
        logger.exception(e)
        return verse


def generate_image_with_prompt(
    cfg: AppConfig,
    state: AppState,
    painter,
    daydream_painter,
    img_prompt: str,
) -> bytes:
    """
    Generate image bytes using the appropriate painter and prompt. May raise on failure.
    """
    # TODO: Improve handling of Stable Core style presets
    if state.daydream:
        if (
            cfg.daydream_image_model == "stableimage"
            and cfg.stable_image_svc == "core"
            and cfg.use_stable_core_presets
        ):
            return daydream_painter.generate_image_data(
                prompt=state.user_prompt,  # Use raw prompt only without base prompt
                core_preset=random.choice(cfg.stable_core_style_presets),
            )
        else:
            return daydream_painter.generate_image_data(prompt=img_prompt)
    else:
        if (
            cfg.image_model == "stableimage"
            and cfg.stable_image_svc == "core"
            and cfg.use_stable_core_presets
        ):
            return painter.generate_image_data(
                prompt=state.user_prompt,  # Use raw prompt only without base prompt
                core_preset=random.choice(cfg.stable_core_style_presets),
            )
        else:
            return painter.generate_image_data(prompt=img_prompt)


def render_creation_display(
    cfg: AppConfig,
    state: AppState,
    verse: str,
    img_bytes: bytes,
    speech_svc: ArtistSpeech,
    user_action: UserAction | None,
    artist_canvas: ArtistCanvas,
    disp_surface: pygame.Surface,
) -> None:
    """
    Load the generated image, render the verse overlay, and update the display.
    """
    verse_lines = [line.strip() for line in verse.split("\n")]
    logger.info(f"Verse: {'/'.join(verse_lines)}")

    img_side = random.choice(["left", "right"])
    finished_phrase = random.choice(cfg.finished_lines)

    if not state.daydream:
        speech_svc.speak_text(text=finished_phrase)
    # Only speak prompt if daydream was manually initiated
    elif user_action == UserAction.DAYDREAM:
        speech_svc.speak_text(text=state.user_prompt, use_cache=False)

    # raw_image.png is a name hint to assist in file format detection, not an actual file on disk
    img = pygame.image.load(io.BytesIO(img_bytes), "raw_image.png")

    if img.get_width() != cfg.img_width or img.get_height() != cfg.img_height:
        logger.warning(
            f"Image has unexpected dimensions {img.get_width()}x{img.get_height()}, expected {cfg.img_width}x{cfg.img_height}"
        )
        img = pygame.transform.smoothscale(img, (cfg.img_width, cfg.img_height))

    creation = ArtistCreation(img, verse_lines, state.user_prompt, state.daydream)
    artist_canvas.render_creation(creation, img_side)
    disp_surface.blit(artist_canvas.surface, (0, 0))
    pygame.display.update()


def save_creation_locally(
    cfg: AppConfig,
    state: AppState,
    disp_surface: pygame.Surface,
) -> str:
    """
    Save the current display as a PNG screenshot. Returns the screenshot filename.
    """
    logger.debug("Saving creation...")
    screenshot_file_name = state.base_file_name + ".png"
    pygame.image.save(disp_surface, os.path.join(cfg.output_dir, screenshot_file_name))
    return screenshot_file_name


def upload_creation_to_storage(
    cfg: AppConfig,
    state: AppState,
    storage: ArtistStorage,
    screenshot_file_name: str,
) -> None:
    """
    Upload the HTML page and PNG screenshot to Azure blob storage.
    """
    logger.debug("Uploading creation...")
    image_url = f"https://{cfg.storage_account}.blob.core.windows.net/{cfg.storage_container}/{screenshot_file_name}"

    html_bytes = io.BytesIO()
    with open(cfg.html_template, "r") as template_file:
        for line in template_file:
            out_line = line.replace("***IMG-URL***", image_url)
            out_line = out_line.replace("***PROMPT***", state.user_prompt)
            out_line = out_line.replace(
                "***GEN-BY***",
                "A.R.T.I.S.T. Daydream" if state.daydream else "User Request",
            )
            out_line = out_line.replace("***TIME***", time.asctime())
            html_bytes.write(out_line.encode())
    html_bytes.seek(0)

    try:
        storage.upload_blob(
            blob_name=state.base_file_name + ".html",
            data=html_bytes.read(),
            content_type="text/html",
        )
    except Exception as e:
        logger.error("Error uploading HTML to blob storage")
        logger.exception(e)

    with open(os.path.join(cfg.output_dir, screenshot_file_name), "rb") as f:
        try:
            storage.upload_blob(
                blob_name=state.base_file_name + ".png",
                data=f.read(),
                content_type="image/png",
            )
        except Exception as e:
            logger.error("Error uploading screenshot to blob storage")
            logger.exception(e)


def update_recents_and_scheduling(cfg: AppConfig, state: AppState) -> None:
    """
    Append the current creation to the recents list and schedule the next auto-daydream.
    """
    logger.debug("Updating recent creations...")
    state.recents.append(
        {
            "base_name": state.base_file_name,
            "prompt": state.user_prompt,
            "daydream": state.daydream,
        }
    )

    if len(state.recents) > cfg.max_recents:
        state.recents = state.recents[-cfg.max_recents :]

    save_recents(state.recents, cfg.recents_file_name)
    state.recent_index = len(state.recents) - 1

    state.next_change_time = time.monotonic() + random.randint(
        cfg.min_daydream_time, cfg.max_daydream_time
    )


def handle_creation_failure(
    cfg: AppConfig,
    speech_svc: ArtistSpeech,
    disp_surface: pygame.Surface,
    status_screen: StatusScreen,
) -> None:
    """
    Display the failure status screen and speak a failure line.
    """
    show_status_screen(
        surface=disp_surface,
        text="Creation failed!",
        status_screen_obj=status_screen,
    )
    speech_svc.speak_text(text=random.choice(cfg.failed_lines))


def run_creation_pipeline(
    user_action: UserAction | None,
    cfg: AppConfig,
    speech_svc: ArtistSpeech,
    audio_recorder: AudioRecorder,
    transcriber: Transcriber,
    ai_artist,
    painter,
    daydream_painter,
    poet,
    critic,
    enhancer,
    moderator: ArtistModerator,
    artist_canvas: ArtistCanvas,
    status_screen: StatusScreen,
    storage: ArtistStorage,
    disp_surface: pygame.Surface,
    state: AppState,
) -> bool:
    """
    Execute a manual creation or daydream creation pipeline.

    Returns True if the outer loop should continue without creating
    (silence detected with no valid audio). Mutates state in place.

    Return value is currently unused.
    """
    if not state.daydream:
        logger.info("=== Starting new creation ===")
        if capture_user_audio(
            cfg,
            speech_svc,
            audio_recorder,
            transcriber,
            disp_surface,
            status_screen,
            state,
        ):
            return True
    else:
        logger.info("=== Starting daydream ===")
        generate_daydream_prompt(
            cfg, state, ai_artist, speech_svc, user_action, disp_surface, status_screen
        )

    state.base_file_name = get_random_string(cfg.file_name_length)
    logger.info(f"Base name: {state.base_file_name}")

    img_prompt = random.choice(cfg.image_base_prompts) + state.user_prompt
    state.previous_user_prompt = state.user_prompt

    can_create = moderator.check_msg(msg=img_prompt)
    creation_failed = False

    if can_create:
        verse = generate_verse(cfg, poet, critic, state)

        use_poem = (state.daydream and cfg.use_poem_as_daydream_prompt) or (
            not state.daydream and cfg.use_poem_as_user_prompt
        )

        if use_poem:
            enhancement_type = (
                cfg.daydream_prompt_enhancement_type
                if state.daydream
                else cfg.user_prompt_enhancement_type
            )
            verse_log = verse.replace("\n", "/")
            logger.info(f"Poem (original): {verse_log}")

            if enhancer is not None and enhancement_type == "llm":
                enhanced = enhance_poem(enhancer, cfg.llm_enhancer_base_prompt, verse)
                logger.info(f"Prompt (enhanced, llm): {enhanced}")
            elif enhancer is not None and enhancement_type == "token":
                enhanced = enhance_poem(enhancer, cfg.token_enhancer_base_prompt, verse)
                logger.info(f"Prompt (enhanced, token): {enhanced}")
            else:
                enhanced = verse

            base_prompt = random.choice(cfg.image_base_prompts)
            logger.info(f"Image base prompt: {base_prompt!r}")
            img_prompt = base_prompt + enhanced
        else:
            base_prompt = random.choice(cfg.image_base_prompts)
            logger.info(f"Image base prompt: {base_prompt!r}")
            img_prompt = base_prompt + state.user_prompt

        try:
            img_bytes = generate_image_with_prompt(
                cfg, state, painter, daydream_painter, img_prompt
            )
        except Exception as e:
            logger.error("Error generating image")
            logger.exception(e)
            creation_failed = True

        if not creation_failed:
            render_creation_display(
                cfg,
                state,
                verse,
                img_bytes,
                speech_svc,
                user_action,
                artist_canvas,
                disp_surface,
            )
            screenshot_file_name = save_creation_locally(cfg, state, disp_surface)
            upload_creation_to_storage(cfg, state, storage, screenshot_file_name)
            update_recents_and_scheduling(cfg, state)

    if not can_create or creation_failed:
        handle_creation_failure(cfg, speech_svc, disp_surface, status_screen)

    return False


def create_painter(model: str, cfg: AppConfig):
    """
    Factory function to create an image generation model instance.
    """
    if model == "sdxl":
        return SDXLCreator(
            api_key=cfg.stability_ai_api_key,
            img_width=cfg.img_width,
            img_height=cfg.img_height,
            steps=cfg.sdxl_steps,
            cfg_scale=cfg.sdxl_cfg_scale,
        )
    elif model == "dalle2":
        return DallE2Creator(
            api_key=cfg.openai_api_key,
            img_width=cfg.img_width,
            img_height=cfg.img_height,
        )
    elif model == "dalle3":
        return DallE3Creator(
            api_key=cfg.openai_api_key,
            img_width=cfg.img_width,
            img_height=cfg.img_height,
            quality=cfg.dalle3_quality,
        )
    elif model == "gpt-image-1":
        return GptImage1Creator(
            api_key=cfg.openai_api_key,
            img_width=cfg.img_width,
            img_height=cfg.img_height,
            quality=cfg.gptimage1_quality,
        )
    elif model == "stableimage":
        return StableImageCreator(
            api_key=cfg.stability_ai_api_key,
            service=cfg.stable_image_svc,
            sd3_model=cfg.sd3_model,
        )
    else:
        raise ValueError(f"Unknown image model: {model}")


def create_chat_character(
    system_prompt: str,
    model: str,
    cfg: AppConfig,
    temperature: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
):
    """
    Factory function to create a chat character using the configured chat service.
    """
    if cfg.chat_service == "anthropic":
        return ClaudeChatCharacter(
            system_prompt=system_prompt,
            model=model,
            api_key=cfg.anthropic_api_key,
            temperature=temperature,
        )
    elif cfg.chat_service == "openai":
        return ChatCharacter(
            system_prompt=system_prompt,
            model=model,
            api_key=cfg.openai_api_key,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
    else:
        raise ValueError(f"Unknown chat service: {cfg.chat_service}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A.R.T.I.S.T.")
    parser.add_argument(
        "--config", default="config.json", help="Path to configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg is None:
        return

    logger.info("*** Starting A.R.T.I.S.T. ***")

    button_config = ButtonConfig(
        generate_button=cfg.generate_button,
        daydream_button=cfg.daydream_button,
        reveal_qr_button=cfg.reveal_qr_button,
        reveal_prompt_button=cfg.reveal_prompt_button,
        shutdown_hold_button=cfg.shutdown_hold_button,
        shutdown_press_button=cfg.shutdown_press_button,
    )

    random.seed()

    logger.debug("Initializing display...")
    disp_surface = init_display(width=cfg.display_width, height=cfg.display_height)

    logger.debug("Initializing joystick...")
    js = init_joystick()

    logger.debug("Initializing speech...")
    speech_svc = ArtistSpeech(
        subscription_key=cfg.azure_speech_key,
        region=cfg.azure_speech_region,
        language=cfg.speech_language,
        gender=cfg.speech_gender,
        voice=cfg.speech_voice,
        cache_dir=cfg.speech_cache_dir,
    )

    logger.debug("Initializing storage...")
    storage = ArtistStorage(
        storage_key=cfg.azure_storage_key,
        storage_account=cfg.storage_account,
        storage_container=cfg.storage_container,
    )

    logger.debug("Initializing audio recorder...")
    audio_recorder = AudioRecorder(
        sample_width=2, channels=1, rate=cfg.input_sample_rate
    )

    logger.debug("Initializing transcriber...")
    transcriber = Transcriber(
        channels=1,
        sample_width=2,
        framerate=cfg.input_sample_rate,
        model=cfg.transcriber_model,
        api_key=cfg.openai_api_key,
    )

    try:
        logger.debug("Initializing autonomous AI artist...")
        ai_artist = create_chat_character(
            system_prompt=cfg.artist_system_prompt,
            model=cfg.artist_chat_model,
            cfg=cfg,
        )

        logger.debug(f"Initializing painter with image model {cfg.image_model}...")
        painter = create_painter(cfg.image_model, cfg)

        logger.debug(
            f"Initializing daydream painter with image model {cfg.daydream_image_model}..."
        )
        daydream_painter = create_painter(cfg.daydream_image_model, cfg)

        logger.debug("Initializing poet...")
        poet = create_chat_character(
            system_prompt=cfg.poet_system_prompt,
            model=cfg.poet_chat_model,
            cfg=cfg,
            temperature=cfg.poet_temperature,
            presence_penalty=cfg.poet_presence_penalty,
            frequency_penalty=cfg.poet_frequency_penalty,
        )

        critic = None
        if cfg.use_critic:
            logger.debug("Initializing critic...")
            critic = create_chat_character(
                system_prompt=cfg.critic_system_prompt,
                model=cfg.critic_chat_model,
                cfg=cfg,
            )

        enhancer = None
        if cfg.enhancer_chat_model and cfg.enhancer_system_prompt:
            logger.debug("Initializing enhancer...")
            enhancer = create_chat_character(
                system_prompt=cfg.enhancer_system_prompt,
                model=cfg.enhancer_chat_model,
                cfg=cfg,
            )
    except ValueError as e:
        print(str(e))
        logger.error(str(e))
        return

    logger.debug("Initializing moderator...")
    moderator = ArtistModerator(api_key=cfg.openai_api_key)

    logger.debug("Initializing artist canvas...")
    artist_canvas = ArtistCanvas(
        width=cfg.display_width,
        height=cfg.display_height,
        horiz_margin=cfg.horiz_margin,
        vert_margin=cfg.vert_margin,
        verse_font_name=cfg.verse_font,
        verse_font_max_size=cfg.verse_font_size,
        verse_line_spacing=cfg.verse_line_spacing,
    )

    logger.debug("Initializing status screen...")
    status_screen = StatusScreen(
        width=cfg.display_width,
        height=cfg.display_height,
        font_name=cfg.status_font,
        heading1_size=cfg.status_heading1_size,
        heading2_size=cfg.status_heading2_size,
        status_size=cfg.status_status_size,
        vert_margin=cfg.vert_margin,
    )

    logger.debug("Loading recent creations...")
    state = AppState(
        recents=load_recents(cfg.recents_file_name),
        next_change_time=time.monotonic()
        + random.randint(cfg.min_daydream_time, cfg.max_daydream_time),
    )

    show_status_screen(
        surface=disp_surface, text="Ready", status_screen_obj=status_screen
    )

    while True:
        # Clear any accumulated events
        _ = pygame.event.get()

        user_action = wait_for_action(
            cfg, js, button_config, speech_svc, disp_surface, artist_canvas, state
        )

        if user_action == UserAction.QUIT:
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

        # Return value is currently unused since creation pipeline handles all user interactions
        # internally, but may be useful for future refactoring
        _ = run_creation_pipeline(
            user_action,
            cfg,
            speech_svc,
            audio_recorder,
            transcriber,
            ai_artist,
            painter,
            daydream_painter,
            poet,
            critic,
            enhancer,
            moderator,
            artist_canvas,
            status_screen,
            storage,
            disp_surface,
            state,
        )


if __name__ == "__main__":
    main()
