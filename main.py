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

import pygame
import qrcode

from pygame.locals import *

from artist_classes import (
    ArtistCanvas,
    ArtistCreation,
    StatusScreen,
    draw_hourglass_indicator,
    get_debug_log_surface,
    get_emotional_state_surface,
    get_prompt_surface,
    show_status_screen,
)
from artist_config import AppConfig, AppState, ButtonConfig, UserAction, load_config
from artist_painters import (
    DallE2Creator,
    DallE3Creator,
    GptImage1Creator,
    SDXLCreator,
    StableImageCreator,
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
            if event.key == K_m:
                return UserAction.SHOW_EMOTIONAL_STATE
            if event.key == K_RIGHT:
                return UserAction.NEXT_RECENT
            if event.key == K_LEFT:
                return UserAction.PREVIOUS_RECENT
            if event.key == K_l:
                return UserAction.SHOW_DEBUG_LOG
        elif js and event.type == pygame.JOYBUTTONDOWN:
            if event.button == button_config.shutdown_press_button:
                if js.get_button(button_config.shutdown_hold_button):
                    return UserAction.QUIT
            if event.button == button_config.debug_press_button:
                if js.get_button(button_config.debug_hold_button):
                    return UserAction.SHOW_DEBUG_LOG
            if event.button == button_config.generate_button:
                return UserAction.NEW
            if event.button == button_config.daydream_button:
                return UserAction.DAYDREAM
            if event.button == button_config.reveal_prompt_button:
                return UserAction.SHOW_PROMPT
            if event.button == button_config.reveal_qr_button:
                return UserAction.SHOW_QR
            if event.button == button_config.emotional_state_press_button:
                if js.get_button(button_config.emotional_state_hold_button):
                    return UserAction.SHOW_EMOTIONAL_STATE
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




def load_recents(recents_file_name: str) -> dict:
    """
    Load recent creations from JSON file if the file exists.
    Returns a dict with 'recents' and 'user_prompts' lists.
    Handles backward compat with old plain-list format.
    """
    try:
        with open(recents_file_name, "r") as recents_file:
            data = json.load(recents_file)
    except FileNotFoundError:
        return {"recents": [], "user_prompts": [], "emotional_state": ""}

    if isinstance(data, list):
        return {"recents": data, "user_prompts": [], "emotional_state": ""}

    return {
        "recents": data.get("recents", []),
        "user_prompts": data.get("user_prompts", []),
        "emotional_state": data.get("emotional_state", ""),
    }


def save_recents(
    recents: list, user_prompts: list, emotional_state: str, recents_file_name: str
) -> None:
    with open(recents_file_name, "w") as recents_file:
        json.dump(
            {
                "recents": recents,
                "user_prompts": user_prompts,
                "emotional_state": emotional_state,
            },
            recents_file,
            indent=4,
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
                speak_buffered_line(cfg, state, speech_svc, "refusal", cfg.daydream_refusal_lines)
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
        elif user_action == UserAction.SHOW_EMOTIONAL_STATE:
            if state.emotional_state:
                es_surface = get_emotional_state_surface(
                    emotional_state=state.emotional_state,
                    width=int(cfg.display_width * 0.75),
                    height=int(cfg.display_height * 0.4),
                    font_name=cfg.prompt_font,
                    font_size=cfg.prompt_font_size,
                )
                x_pos = int((cfg.display_width - es_surface.get_width()) / 2)
                y_pos = int((cfg.display_height - es_surface.get_height()) / 2)
                disp_surface.blit(es_surface, (x_pos, y_pos))
                pygame.display.update()
                time.sleep(cfg.prompt_display_time)
                disp_surface.blit(artist_canvas.surface, (0, 0))
                pygame.display.update()
        elif user_action == UserAction.SHOW_DEBUG_LOG:
            debug_surface = get_debug_log_surface(
                log_file="artist.log",
                width=cfg.display_width,
                height=cfg.display_height,
                font_name=cfg.debug_font,
                font_size=cfg.debug_font_size,
            )
            disp_surface.blit(debug_surface, (0, 0))
            pygame.display.update()
            time.sleep(cfg.debug_display_time)
            disp_surface.blit(artist_canvas.surface, (0, 0))
            pygame.display.update()
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
    welcome_fallbacks = [f"{w} {l}" for w in cfg.welcome_words for l in cfg.welcome_lines]
    speak_buffered_line(cfg, state, speech_svc, "welcome", welcome_fallbacks)
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
            speak_buffered_line(cfg, state, speech_svc, "working", cfg.working_lines)
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


def get_overused_topics(recents: list, repeat_limit: int) -> list[str]:
    """
    Return topics that appear >= repeat_limit times across daydream entries in recents.
    """
    counts: dict[str, int] = {}
    for entry in recents:
        if not entry.get("daydream"):
            continue
        topics = entry.get("topics")
        if not topics:
            continue
        for topic in topics.get("subjects", []) + topics.get("settings", []):
            key = topic.lower()
            counts[key] = counts.get(key, 0) + 1
    return [topic for topic, count in counts.items() if count >= repeat_limit]


def extract_daydream_topics(prompt: str, archivist) -> dict | None:
    """
    Use an LLM to extract subjects and settings from a daydream prompt.
    Returns {"subjects": [...], "settings": [...]} or None on failure.
    """
    archivist.reset()
    try:
        response = archivist.get_chat_response(message=prompt)
        raw = response.content.strip()
        logger.debug(f"Raw topic extraction response: {raw}")
        # Strip markdown code fences if present (e.g. ```json ... ```)
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        topics = json.loads(raw)
        if isinstance(topics.get("subjects"), list) and isinstance(
            topics.get("settings"), list
        ):
            return topics
        logger.warning(f"Unexpected topic extraction response format: {topics}")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse topic extraction response as JSON: {e}")
    except Exception as e:
        logger.warning(f"Topic extraction failed: {e}")
    return None


def generate_daydream_prompt(
    cfg: AppConfig,
    state: AppState,
    ai_artist,
    archivist,
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
        speak_buffered_line(cfg, state, speech_svc, "daydream", cfg.daydream_lines)

    if len(state.recents) >= cfg.num_recents_for_daydream:
        daydream_prompt = " , ".join(
            [r["prompt"] for r in state.recents[-cfg.num_recents_for_daydream :]]
        )
    elif len(state.recents) > 0:
        daydream_prompt = " , ".join([r["prompt"] for r in state.recents])
    else:
        daydream_prompt = " something completely random."

    if cfg.enable_emotion_chip and state.emotional_state:
        full_prompt = (
            f"Your current emotional state is {state.emotional_state}. This emotional state should influence the style, tone and content of your response. "
            + cfg.artist_base_prompt
            + " "
            + daydream_prompt
        )
    else:
        full_prompt = cfg.artist_base_prompt + " " + daydream_prompt

    if cfg.enable_daydream_topics:
        overused = get_overused_topics(state.recents, cfg.daydream_topic_repeat_limit)
        if overused:
            logger.info(f"Excluding overused daydream topics: {', '.join(overused)}")
            full_prompt += f" Avoid using these overused subjects and settings: {', '.join(overused)}."
        else:
            logger.debug("No overused daydream topics to exclude")

    logger.debug(f"Daydreaming based on: {daydream_prompt}")
    state.user_prompt = ai_artist.get_chat_response(message=full_prompt).content
    logger.info(f"Daydreamed: {state.user_prompt}")

    state.pending_daydream_topics = None
    if cfg.enable_daydream_topics and archivist is not None:
        state.pending_daydream_topics = extract_daydream_topics(
            state.user_prompt, archivist
        )
        if state.pending_daydream_topics:
            logger.info(
                f"Extracted daydream topics — subjects: {state.pending_daydream_topics.get('subjects', [])}, "
                f"settings: {state.pending_daydream_topics.get('settings', [])}"
            )
        else:
            logger.debug("Topic extraction returned no results")


def generate_verse(cfg: AppConfig, poet, critic, state: AppState, base_prompt: str | None = None) -> str:
    """
    Generate a verse using the poet, optionally with critic selection.
    """
    effective_base_prompt = base_prompt if base_prompt is not None else cfg.verse_base_prompt
    if cfg.use_critic:
        logger.debug("Getting best verse...")
        return get_best_verse(
            poet=poet,
            critic=critic,
            base_prompt=effective_base_prompt,
            user_prompt=state.user_prompt,
            num_verses=cfg.num_verses,
        )
    else:
        logger.debug("Getting one verse...")
        return get_one_verse(
            poet=poet,
            base_prompt=effective_base_prompt,
            user_prompt=state.user_prompt,
        )


def build_visionary_prompt(visionary, base_prompt: str | None, verse: str) -> str:
    """
    Use the visionary to interpret a poem into an image prompt optimized for image models.
    Falls back to the original verse if the visionary fails.
    """
    visionary.reset()

    try:
        return visionary.get_chat_response((base_prompt or "") + verse).content
    except Exception as e:
        logger.error("Error building visionary prompt")
        logger.exception(e)
        return verse


def generate_image_with_prompt(
    state: AppState,
    painter,
    daydream_painter,
    img_prompt: str,
) -> bytes:
    """
    Generate image bytes using the appropriate painter and prompt. May raise on failure.
    """
    if state.daydream:
        return daydream_painter.generate_image_data(prompt=img_prompt)
    else:
        return painter.generate_image_data(prompt=img_prompt)


def draw_hourglass_indicator(
    disp_surface: pygame.Surface,
    poem_side: str,
    display_width: int,
    display_height: int,
) -> None:
    """
    Draw a small hourglass icon in the lower corner on the poem side to signal
    that background processing is underway and input is temporarily blocked.
    """
    size = 128
    margin = 16
    pad = 14

    x = (display_width - size - margin) if poem_side == "right" else margin
    y = display_height - size - margin

    pygame.draw.rect(disp_surface, (20, 20, 20), (x, y, size, size))
    cx = x + size // 2
    cy = y + size // 2
    neck_w = 5   # half-width of the waist in pixels
    cap_h = 9    # height of the flat top/bottom caps
    color = (220, 220, 220)
    dark = (20, 20, 20)

    # Single 6-point polygon: two cones joined by a narrow waist
    body_top = y + pad + cap_h
    body_bot = y + size - pad - cap_h
    pygame.draw.polygon(disp_surface, color, [
        (x + pad,        body_top),
        (x + size - pad, body_top),
        (cx + neck_w,    cy),
        (x + size - pad, body_bot),
        (x + pad,        body_bot),
        (cx - neck_w,    cy),
    ])

    # Flat caps at top and bottom — make it read as a physical container
    pygame.draw.rect(disp_surface, color, (x + pad, y + pad, size - 2 * pad, cap_h + 2))
    pygame.draw.rect(disp_surface, color, (x + pad, y + size - pad - cap_h - 2, size - 2 * pad, cap_h + 2))

    # Thin dark line separating each cap from its cone, for depth
    pygame.draw.rect(disp_surface, dark, (x + pad, y + pad + cap_h, size - 2 * pad, 3))
    pygame.draw.rect(disp_surface, dark, (x + pad, y + size - pad - cap_h - 3, size - 2 * pad, 3))

    pygame.draw.rect(disp_surface, color, (x, y, size, size), 2)
    pygame.display.update()


def render_creation_display(
    cfg: AppConfig,
    state: AppState,
    verse: str,
    img_bytes: bytes,
    speech_svc: ArtistSpeech,
    user_action: UserAction | None,
    artist_canvas: ArtistCanvas,
    disp_surface: pygame.Surface,
) -> str:
    """
    Load the generated image, render the verse overlay, and update the display.
    Returns img_side ("left" or "right") so callers know which side the poem is on.
    """
    verse_lines = [line.strip() for line in verse.split("\n")]
    logger.info(f"Verse: {'/'.join(verse_lines)}")

    img_side = random.choice(["left", "right"])

    if not state.daydream:
        speak_buffered_line(cfg, state, speech_svc, "finished", cfg.finished_lines)
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
    return img_side


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
    entry = {
        "base_name": state.base_file_name,
        "prompt": state.user_prompt,
        "daydream": state.daydream,
    }
    if state.daydream and state.pending_daydream_topics:
        entry["topics"] = state.pending_daydream_topics
        logger.debug(f"Saving daydream topics to recents: {state.pending_daydream_topics}")
    state.pending_daydream_topics = None
    state.recents.append(entry)

    if len(state.recents) > cfg.max_recents:
        state.recents = state.recents[-cfg.max_recents :]

    if not state.daydream:
        state.user_prompts.append({"prompt": state.user_prompt})

    if len(state.user_prompts) > cfg.num_user_prompts_for_emotions:
        state.user_prompts = state.user_prompts[-cfg.num_user_prompts_for_emotions :]

    save_recents(
        state.recents, state.user_prompts, state.emotional_state, cfg.recents_file_name
    )
    state.recent_index = len(state.recents) - 1

    state.next_change_time = time.monotonic() + random.randint(
        cfg.min_daydream_time, cfg.max_daydream_time
    )


def generate_emotional_state(cfg: AppConfig, state: AppState, emotion_chip) -> None:
    """
    Use the emotion_chip chat character to generate an emotional state from recent
    user prompts and save it to state.emotional_state.
    """
    if not state.user_prompts:
        return

    prompt_list = ", ".join(entry["prompt"] for entry in state.user_prompts)

    emotion_chip.reset()
    try:
        state.emotional_state = emotion_chip.get_chat_response(
            message=(cfg.emotion_chip_base_prompt or "") + " " + prompt_list
        ).content
        logger.info(f"Emotional state: {state.emotional_state}")
    except Exception as e:
        logger.error("Error generating emotional state")
        logger.exception(e)


def drift_emotional_state(cfg: AppConfig, state: AppState, emotion_chip) -> None:
    """
    Gently drift the emotional state based on recent daydream themes when no
    user prompts have been received. The current emotional state is provided as
    context so the model can evolve from it, while the base prompt still instructs
    it to output a state description rather than a description of change.
    """
    recent_daydream_prompts = [
        r["prompt"] for r in state.recents if r["daydream"]
    ][-cfg.num_recents_for_daydream:]

    if not recent_daydream_prompts:
        return

    current_state_context = (
        f"Your current emotional state is: {state.emotional_state}. "
        if state.emotional_state
        else ""
    )
    message = (
        current_state_context
        + (cfg.emotion_chip_base_prompt or "")
        + " "
        + ", ".join(recent_daydream_prompts)
    )

    emotion_chip.reset()
    try:
        state.emotional_state = emotion_chip.get_chat_response(message=message).content
        logger.info(f"Emotional state (drifted): {state.emotional_state}")
    except Exception as e:
        logger.error("Error drifting emotional state")
        logger.exception(e)


def generate_speech_line_buffer(
    cfg: AppConfig,
    state: AppState,
    raconteur,
    speech_svc: ArtistSpeech,
) -> None:
    """
    Use the raconteur character to generate speech lines for any empty buffer
    categories and synthesize each via TTS (no file cache).

    If the emotional state has changed since the last fill, clears all existing
    buffered lines first so stale lines are replaced. Skips categories that
    already have lines when the emotional state is unchanged.
    """
    valid_categories = ["welcome", "working", "daydream", "finished", "failed", "refusal"]

    emotional_state_changed = state.emotional_state != state.speech_line_buffer_emotional_state
    if emotional_state_changed:
        logger.info("Raconteur: emotional state changed — clearing buffer and regenerating all lines...")
        state.speech_line_buffer.clear()
        state.speech_line_buffer_emotional_state = state.emotional_state

    categories_needing_lines = [c for c in valid_categories if not state.speech_line_buffer.get(c)]
    if not categories_needing_lines:
        logger.debug("Raconteur: all categories have buffered lines — skipping generation")
        return

    logger.info(f"Raconteur: generating lines for: {categories_needing_lines}...")

    if cfg.enable_emotion_chip and state.emotional_state:
        full_prompt = (
            f"Your current emotional state is {state.emotional_state}. This emotional state should influence the style, tone and content of your response. "
            + (cfg.raconteur_base_prompt or "")
        )
    else:
        full_prompt = cfg.raconteur_base_prompt or ""

    raconteur.reset()
    try:
        response = raconteur.get_chat_response(message=full_prompt)
        raw = response.content
        logger.debug(f"Raconteur raw response: {raw}")
        lines = json.loads(raw)
    except Exception as e:
        logger.error("Raconteur: error generating speech lines")
        logger.exception(e)
        return

    for category, text in lines.items():
        if category not in valid_categories:
            logger.warning(f"Raconteur: unexpected category '{category}' — skipping")
            continue
        if state.speech_line_buffer.get(category):
            logger.debug(f"Raconteur: '{category}' already has lines — skipping")
            continue
        try:
            audio_data = speech_svc.synthesize_text(text)
            state.speech_line_buffer[category] = [audio_data]
            logger.debug(f"Raconteur: buffered '{category}': {text}")
        except Exception as e:
            logger.error(f"Raconteur: error synthesizing '{category}'")
            logger.exception(e)


def speak_buffered_line(
    cfg: AppConfig,
    state: AppState,
    speech_svc: ArtistSpeech,
    category: str,
    fallback: list,
) -> None:
    """
    Speak a pre-synthesized line from the in-memory buffer if available;
    otherwise fall back to speaking a random line from fallback with TTS caching.
    """
    if cfg.dynamic_speech_lines and state.speech_line_buffer.get(category):
        audio_data = state.speech_line_buffer[category].pop(0)
        speech_svc.play_audio(audio_data)
    else:
        speech_svc.speak_text(text=random.choice(fallback))


def handle_creation_failure(
    cfg: AppConfig,
    state: AppState,
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
    speak_buffered_line(cfg, state, speech_svc, "failed", cfg.failed_lines)


def run_creation_pipeline(
    user_action: UserAction | None,
    cfg: AppConfig,
    speech_svc: ArtistSpeech,
    audio_recorder: AudioRecorder,
    transcriber: Transcriber,
    ai_artist,
    archivist,
    painter,
    daydream_painter,
    poet,
    critic,
    visionary,
    emotion_chip,
    raconteur,
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
            cfg, state, ai_artist, archivist, speech_svc, user_action, disp_surface, status_screen
        )

    state.base_file_name = get_random_string(cfg.file_name_length)
    logger.info(f"Base name: {state.base_file_name}")

    img_prompt = state.user_prompt
    state.previous_user_prompt = state.user_prompt

    can_create = moderator.check_msg(msg=img_prompt)
    creation_failed = False

    if can_create:
        if not state.daydream and cfg.enable_emotion_chip and state.emotional_state:
            logger.info(f"Generating verse with emotional state: {state.emotional_state}")
            verse_base_prompt = (
                f"Your current emotional state is {state.emotional_state}. This emotional state should influence the style, tone and content of your response. "
                + cfg.verse_base_prompt
            )
        else:
            verse_base_prompt = None
        verse = generate_verse(cfg, poet, critic, state, base_prompt=verse_base_prompt)

        enhancement_type = (
            cfg.daydream_prompt_enhancement_type
            if state.daydream
            else cfg.user_prompt_enhancement_type
        )
        verse_log = verse.replace("\n", "/")
        logger.info(f"Poem (original): {verse_log}")

        if enhancement_type == "llm":
            img_prompt = build_visionary_prompt(visionary, cfg.llm_visionary_base_prompt, verse)
            logger.info(f"Prompt (enhanced, llm): {img_prompt}")
        elif enhancement_type == "clip":
            img_prompt = build_visionary_prompt(visionary, cfg.clip_visionary_base_prompt, verse)
            logger.info(f"Prompt (enhanced, clip): {img_prompt}")
        else:
            img_prompt = verse
            logger.info(f"Prompt (unenhanced): {img_prompt}")

        try:
            img_bytes = generate_image_with_prompt(
                state, painter, daydream_painter, img_prompt
            )
        except Exception as e:
            logger.error("Error generating image")
            logger.exception(e)
            creation_failed = True

        if not creation_failed:
            img_side = render_creation_display(
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
            if emotion_chip:
                if not state.daydream:
                    state.daydreams_since_user_prompt = 0
                    generate_emotional_state(cfg, state, emotion_chip)
                    save_recents(
                        state.recents,
                        state.user_prompts,
                        state.emotional_state,
                        cfg.recents_file_name,
                    )
                else:
                    state.daydreams_since_user_prompt += 1
                    if state.daydreams_since_user_prompt >= cfg.emotion_drift_interval:
                        state.daydreams_since_user_prompt = 0
                        drift_emotional_state(cfg, state, emotion_chip)
                        save_recents(
                            state.recents,
                            state.user_prompts,
                            state.emotional_state,
                            cfg.recents_file_name,
                        )

            if raconteur:
                poem_side = "right" if img_side == "left" else "left"
                draw_hourglass_indicator(disp_surface, poem_side, cfg.display_width, cfg.display_height)
                generate_speech_line_buffer(cfg, state, raconteur, speech_svc)
                disp_surface.blit(artist_canvas.surface, (0, 0))
                pygame.display.update()

    if not can_create or creation_failed:
        handle_creation_failure(cfg, state, speech_svc, disp_surface, status_screen)
        if raconteur:
            generate_speech_line_buffer(cfg, state, raconteur, speech_svc)

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
        emotional_state_hold_button=cfg.emotional_state_hold_button,
        emotional_state_press_button=cfg.emotional_state_press_button,
        shutdown_hold_button=cfg.shutdown_hold_button,
        shutdown_press_button=cfg.shutdown_press_button,
        debug_hold_button=cfg.debug_hold_button,
        debug_press_button=cfg.debug_press_button,
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

        logger.debug("Initializing visionary...")
        visionary = create_chat_character(
            system_prompt=cfg.visionary_system_prompt,
            model=cfg.visionary_chat_model,
            cfg=cfg,
        )

        emotion_chip = None
        if cfg.enable_emotion_chip:
            if (
                not cfg.emotion_chip_system_prompt
                or not cfg.emotion_chip_base_prompt
                or not cfg.emotion_chip_chat_model
            ):
                raise ValueError(
                    "emotion_chip_system_prompt, emotion_chip_base_prompt, and emotion_chip_chat_model "
                    "must all be set in config when enable_emotion_chip is true."
                )
            logger.debug("Initializing emotion chip...")
            emotion_chip = create_chat_character(
                system_prompt=cfg.emotion_chip_system_prompt,
                model=cfg.emotion_chip_chat_model,
                cfg=cfg,
            )

        raconteur = None
        if cfg.dynamic_speech_lines:
            if (
                not cfg.raconteur_system_prompt
                or not cfg.raconteur_base_prompt
                or not cfg.raconteur_chat_model
            ):
                raise ValueError(
                    "raconteur_system_prompt, raconteur_base_prompt, and raconteur_chat_model "
                    "must all be set in config when dynamic_speech_lines is true."
                )
            logger.debug("Initializing raconteur...")
            raconteur = create_chat_character(
                system_prompt=cfg.raconteur_system_prompt,
                model=cfg.raconteur_chat_model,
                cfg=cfg,
            )

        archivist = None
        if cfg.enable_daydream_topics:
            if not cfg.archivist_system_prompt or not cfg.archivist_chat_model:
                raise ValueError(
                    "archivist_system_prompt and archivist_chat_model "
                    "must both be set in config when enable_daydream_topics is true."
                )
            logger.debug("Initializing archivist...")
            archivist = create_chat_character(
                system_prompt=cfg.archivist_system_prompt,
                model=cfg.archivist_chat_model,
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
    recents_data = load_recents(cfg.recents_file_name)
    state = AppState(
        recents=recents_data["recents"],
        user_prompts=recents_data["user_prompts"],
        emotional_state=recents_data["emotional_state"],
        next_change_time=time.monotonic()
        + random.randint(cfg.min_daydream_time, cfg.max_daydream_time),
    )

    if cfg.dynamic_speech_lines and raconteur:
        show_status_screen(surface=disp_surface, text="Waking up...", status_screen_obj=status_screen)
        generate_speech_line_buffer(cfg, state, raconteur, speech_svc)

    show_status_screen(
        surface=disp_surface, text="Ready", status_screen_obj=status_screen
    )

    while True:
        # Clear any accumulated events
        _ = pygame.event.get()

        try:
            user_action = wait_for_action(
                cfg, js, button_config, speech_svc, disp_surface, artist_canvas, state
            )

            if user_action == UserAction.QUIT:
                logger.info("*** A.R.T.I.S.T. is shutting down. ***")
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
                archivist,
                painter,
                daydream_painter,
                poet,
                critic,
                visionary,
                emotion_chip,
                raconteur,
                moderator,
                artist_canvas,
                status_screen,
                storage,
                disp_surface,
                state,
            )
        except Exception as e:
            logger.error("Unhandled exception in main loop — recovering")
            logger.exception(e)
            show_status_screen(
                surface=disp_surface, text="Ready", status_screen_obj=status_screen
            )


if __name__ == "__main__":
    main()
