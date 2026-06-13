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

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum

import pygame

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


@dataclass
class CharacterConfig:
    """
    Per-character settings for a chat character role.
    """

    service: str
    model: str
    system_prompt: str
    options: dict = field(default_factory=dict)
    base_prompt: str = ""
    llm_base_prompt: str = ""
    clip_base_prompt: str = ""


@dataclass
class ButtonConfig:
    """
    Game controller button mappings for the check_for_event function.
    """

    generate_button: int
    daydream_button: int
    reveal_qr_button: int
    reveal_prompt_button: int
    emotional_state_hold_button: int
    emotional_state_press_button: int
    shutdown_hold_button: int
    shutdown_press_button: int
    debug_hold_button: int
    debug_press_button: int


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
    SHOW_EMOTIONAL_STATE = 9
    SHOW_DEBUG_LOG = 10


@dataclass
class AppConfig:
    """
    All configuration values loaded from the config file and environment variables.
    """

    # Chat characters
    artist: CharacterConfig
    poet: CharacterConfig
    visionary: CharacterConfig

    # Image generation
    image_model: str
    daydream_image_model: str
    img_width: int
    img_height: int

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
    debug_font: str
    debug_font_size: int
    debug_display_time: float

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
    emotional_state_hold_button: int
    emotional_state_press_button: int
    shutdown_hold_button: int
    shutdown_press_button: int
    debug_hold_button: int
    debug_press_button: int

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
    num_verses: int = 3
    use_critic: bool = False
    critic: CharacterConfig | None = None
    user_prompt_enhancement_type: str | None = None
    daydream_prompt_enhancement_type: str | None = None
    sdxl_steps: int | None = None
    sdxl_cfg_scale: float | None = None
    gptimage1_quality: str | None = None
    stable_image_svc: str | None = None
    sd3_model: str | None = None
    num_user_prompts_for_emotions: int = 5
    enable_emotion_chip: bool = False
    emotion_chip: CharacterConfig | None = None
    emotion_drift_interval: int = 3
    anthropic_api_key: str | None = None
    stability_ai_api_key: str | None = None
    openrouter_api_key: str | None = None
    fal_api_key: str | None = None
    dynamic_speech_lines: bool = False
    raconteur: CharacterConfig | None = None
    enable_daydream_topics: bool = True
    archivist: CharacterConfig | None = None
    daydream_topic_repeat_limit: int = 3
    disk_space_warn_pct: float = 10.0
    disk_space_target_pct: float = 20.0


@dataclass
class AppState:
    """
    Mutable runtime state for the main application loop.
    """

    js: pygame.joystick.JoystickType | None = None
    daydream: bool = False
    manual_daydream_timestamps: list = field(default_factory=list)
    base_file_name: str | None = None
    user_prompt: str = ""
    previous_user_prompt: str = ""
    recents: list = field(default_factory=list)
    user_prompts: list = field(default_factory=list)
    emotional_state: str = ""
    recent_index: int = 0
    next_change_time: float = 0.0
    daydreams_since_user_prompt: int = 0
    speech_line_buffer: dict = field(default_factory=dict)
    speech_line_buffer_emotional_state: str = ""
    pending_daydream_topics: dict | None = None


def _parse_character_config(data: dict, name: str) -> CharacterConfig | None:
    """Parse a character config dict, returning None and printing an error if required keys are missing."""
    for key in ("service", "model", "system_prompt"):
        if key not in data:
            print(f"Missing required key '{key}' in character config '{name}'.")
            return None
    return CharacterConfig(
        service=data["service"],
        model=data["model"],
        system_prompt=data["system_prompt"],
        options=data.get("options", {}),
        base_prompt=data.get("base_prompt", ""),
        llm_base_prompt=data.get("llm_base_prompt", ""),
        clip_base_prompt=data.get("clip_base_prompt", ""),
    )


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
        "artist",
        "poet",
        "visionary",
        "image_model",
        "daydream_image_model",
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
        "emotional_state_hold_button",
        "emotional_state_press_button",
        "shutdown_hold_button",
        "shutdown_press_button",
        "debug_hold_button",
        "debug_press_button",
        "debug_font",
        "debug_font_size",
        "debug_display_time",
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

    # Parse required character configs
    artist_cfg = _parse_character_config(config["artist"], "artist")
    poet_cfg = _parse_character_config(config["poet"], "poet")
    visionary_cfg = _parse_character_config(config["visionary"], "visionary")
    if not artist_cfg or not poet_cfg or not visionary_cfg:
        return None

    # Parse optional character configs
    critic_cfg = None
    if "critic" in config:
        critic_cfg = _parse_character_config(config["critic"], "critic")
        if not critic_cfg:
            return None

    emotion_chip_cfg = None
    if "emotion_chip" in config:
        emotion_chip_cfg = _parse_character_config(config["emotion_chip"], "emotion_chip")
        if not emotion_chip_cfg:
            return None

    raconteur_cfg = None
    if "raconteur" in config:
        raconteur_cfg = _parse_character_config(config["raconteur"], "raconteur")
        if not raconteur_cfg:
            return None

    archivist_cfg = None
    if "archivist" in config:
        archivist_cfg = _parse_character_config(config["archivist"], "archivist")
        if not archivist_cfg:
            return None

    # Collect all services in use to determine which API keys are required
    all_chars = [c for c in [artist_cfg, poet_cfg, visionary_cfg, critic_cfg,
                              emotion_chip_cfg, raconteur_cfg, archivist_cfg] if c]
    services_in_use = {c.service for c in all_chars}

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
    if "anthropic" in services_in_use:
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key is None:
            print("Please set ANTHROPIC_API_KEY environment variable for Anthropic API key.")
            return None

    openrouter_api_key = None
    if "openrouter" in services_in_use:
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_api_key is None:
            print("Please set OPENROUTER_API_KEY environment variable for OpenRouter API key.")
            return None

    fal_api_key = None
    if config["image_model"].startswith("fal-ai/") or config["daydream_image_model"].startswith("fal-ai/"):
        fal_api_key = os.environ.get("FAL_API_KEY")
        if fal_api_key is None:
            print("Please set FAL_API_KEY environment variable for Fal.ai API key.")
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
        artist=artist_cfg,
        poet=poet_cfg,
        visionary=visionary_cfg,
        num_verses=config.get("num_verses", 3),
        use_critic=config.get("use_critic", False),
        critic=critic_cfg,
        image_model=config["image_model"],
        daydream_image_model=config["daydream_image_model"],
        img_width=config["img_width"],
        img_height=config["img_height"],
        user_prompt_enhancement_type=config.get("user_prompt_enhancement_type"),
        daydream_prompt_enhancement_type=config.get("daydream_prompt_enhancement_type"),
        sdxl_steps=config.get("sdxl_steps"),
        sdxl_cfg_scale=config.get("sdxl_cfg_scale"),
        gptimage1_quality=config.get("gptimage1_quality"),
        stable_image_svc=config.get("stableimage_svc"),
        sd3_model=config.get("sd3_model"),
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
        num_user_prompts_for_emotions=config.get("num_user_prompts_for_emotions", 5),
        enable_emotion_chip=config.get("enable_emotion_chip", False),
        emotion_chip=emotion_chip_cfg,
        emotion_drift_interval=config.get("emotion_drift_interval", 3),
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
        emotional_state_hold_button=config["emotional_state_hold_button"],
        emotional_state_press_button=config["emotional_state_press_button"],
        shutdown_hold_button=config["shutdown_hold_button"],
        shutdown_press_button=config["shutdown_press_button"],
        debug_hold_button=config["debug_hold_button"],
        debug_press_button=config["debug_press_button"],
        debug_font=config["debug_font"],
        debug_font_size=config["debug_font_size"],
        debug_display_time=config["debug_display_time"],
        welcome_words=config["welcome_words"],
        welcome_lines=config["welcome_lines"],
        daydream_lines=config["daydream_lines"],
        working_lines=config["working_lines"],
        finished_lines=config["finished_lines"],
        failed_lines=config["failed_lines"],
        daydream_refusal_lines=config["daydream_refusal_lines"],
        dynamic_speech_lines=config.get("dynamic_speech_lines", False),
        raconteur=raconteur_cfg,
        enable_daydream_topics=config.get("enable_daydream_topics", True),
        archivist=archivist_cfg,
        daydream_topic_repeat_limit=config.get("daydream_topic_repeat_limit", 3),
        disk_space_warn_pct=config.get("disk_space_warn_percentage", 10.0),
        disk_space_target_pct=config.get("disk_space_target_percentage", 20.0),
        openai_api_key=openai_api_key,
        azure_speech_region=azure_speech_region,
        azure_speech_key=azure_speech_key,
        azure_storage_key=azure_storage_key,
        anthropic_api_key=anthropic_api_key,
        stability_ai_api_key=stability_ai_api_key,
        openrouter_api_key=openrouter_api_key,
        fal_api_key=fal_api_key,
    )
