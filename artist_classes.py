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

import pygame

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


class ArtistCreation:
    """
    Class representing a full "creation" by the A.R.T.I.S.T. system, i.e., the image
    and its corresponding verse.
    """

    def __init__(
        self,
        img: pygame.Surface,
        verse_lines: list[str],
        prompt: str,
        is_daydream: bool,
    ) -> None:
        self.img = img
        self.verse_lines = verse_lines
        self.prompt = prompt
        self.is_daydream = is_daydream


class ArtistCanvas:
    """
    Class representing the visible surface on which the ArtistCreation object
    will be rendered.
    """

    def __init__(
        self,
        width: int,
        height: int,
        horiz_margin: int,
        vert_margin: int,
        verse_font_name: str,
        verse_font_max_size: int,
        verse_line_spacing: int,
    ) -> None:
        self._width = width
        self._height = height

        self._horiz_margin = horiz_margin
        self._vert_margin = vert_margin

        self._verse_font_name = verse_font_name
        self._verse_font_max_size = verse_font_max_size
        self._verse_line_spacing = verse_line_spacing

        self._surface = pygame.Surface(size=(width, height))

    def _get_verse_font_size(self, verse_lines: list[str], max_verse_width: int) -> int:
        font_obj = pygame.font.SysFont(self._verse_font_name, self._verse_font_max_size)
        longest_line = ""
        longest_line_size = 0

        # Need to check pizel size of each line to account for
        # proprtional fonts. Assumes that size scales linearly.
        for line in verse_lines:
            text_size = font_obj.size(line)
            if text_size[0] > longest_line_size:
                longest_line_size = text_size[0]
                longest_line = line

        font_size = self._verse_font_max_size
        will_fit = False

        while not will_fit:
            font_obj = pygame.font.SysFont(self._verse_font_name, font_size)

            text_size = font_obj.size(longest_line)

            if text_size[0] < max_verse_width:
                will_fit = True
            else:
                font_size -= 2

        return font_size

    def _get_verse_total_height(
        self, verse_lines: list[str], verse_font_size: int
    ) -> int:
        font_obj = pygame.font.SysFont(self._verse_font_name, verse_font_size)

        total_height = 0

        for line in verse_lines:
            text_size = font_obj.size(line)

            total_height += text_size[1]
            total_height += self._verse_line_spacing

        total_height -= self._verse_line_spacing

        return total_height

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def clear(self) -> None:
        self._surface.fill(color=pygame.Color("black"))

    def render_creation(self, creation: ArtistCreation, img_side: str) -> None:
        self.clear()

        img_width = creation.img.get_width()

        if img_side.lower() == "left":
            img_x = self._horiz_margin
            verse_x = self._horiz_margin + img_width + self._horiz_margin
        elif img_side.lower() == "right":
            img_x = self._width - self._horiz_margin - img_width
            verse_x = self._horiz_margin
        else:
            raise ValueError("img_side must be either 'left' or 'right'")

        # Draw the image
        self._surface.blit(source=creation.img, dest=(img_x, self._vert_margin))

        max_verse_width = (self._width - img_width) - (self._horiz_margin * 3)
        verse_font_size = self._get_verse_font_size(
            creation.verse_lines, max_verse_width
        )

        total_height = self._get_verse_total_height(
            creation.verse_lines, verse_font_size
        )
        offset = -total_height // 2

        font_obj = pygame.font.SysFont(self._verse_font_name, verse_font_size)

        for line in creation.verse_lines:
            text_surface = font_obj.render(line, True, pygame.Color("white"))
            self._surface.blit(
                source=text_surface, dest=(verse_x, (self._height // 2) + offset)
            )

            offset += int(total_height / len(creation.verse_lines))


class StatusScreen:
    """
    Class representing the status screen displayed when A.R.T.I.S.T. is
    waiting for input or generating a new creation.
    """

    def __init__(
        self,
        width: int,
        height: int,
        font_name: str,
        heading1_size: int,
        heading2_size: int,
        status_size: int,
        vert_margin: int,
    ) -> None:
        self._width = width
        self._height = height
        self._font_name = font_name
        self._heading1_size = heading1_size
        self._heading2_size = heading2_size
        self._status_size = status_size
        self._vert_margin = vert_margin

        self._surface = pygame.Surface(size=(width, height))

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def render_status(self, text: str) -> None:
        self._surface.fill(pygame.Color("black"))

        font = pygame.font.SysFont(self._font_name, self._heading1_size)
        heading1 = "A.R.T.I.S.T."
        x_pos = int(self._surface.get_width() / 2 - font.size(heading1)[0] / 2)
        y_pos = self._vert_margin
        text_surface = font.render(heading1, True, pygame.Color("white"))
        self._surface.blit(text_surface, (x_pos, y_pos))

        heading1_height = font.size(heading1)[1]

        font = pygame.font.SysFont(self._font_name, self._heading2_size)
        heading2 = "Autonomous Reflective Transformative Intelligence with Sentient Tendencies"
        x_pos = int(self._surface.get_width() / 2 - font.size(heading2)[0] / 2)
        y_pos += heading1_height
        text_surface = font.render(heading2, True, pygame.Color("white"))
        self._surface.blit(text_surface, (x_pos, y_pos))

        font = pygame.font.SysFont(self._font_name, self._status_size)
        x_pos = int(self._surface.get_width() / 2 - font.size(text)[0] / 2)
        y_pos = int(self._surface.get_height() / 2 - font.size(text)[1] / 2)
        text_surface = font.render(text, True, pygame.Color("white"))
        self._surface.blit(text_surface, (x_pos, y_pos))


def update_display(
    display_surface: pygame.Surface, content_surface: pygame.Surface
) -> None:
    display_surface.blit(content_surface, (0, 0))
    pygame.display.update()


def show_status_screen(
    surface: pygame.Surface, text: str, status_screen_obj: StatusScreen
) -> None:
    status_screen_obj.render_status(text)
    update_display(surface, status_screen_obj.surface)


def get_prompt_surface(
    prompt: str,
    prompt_source: str,
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    margin_size: int = 10,
) -> pygame.Surface:
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


def get_emotional_state_surface(
    emotional_state: str,
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    margin_size: int = 10,
) -> pygame.Surface:
    surface = pygame.Surface((width, height))
    surface.fill(pygame.Color("purple"))

    text_surface = pygame.Surface(
        (width - (margin_size * 2), height - (margin_size * 2))
    )
    text_surface.fill(pygame.Color("black"))

    text_subsurface = pygame.Surface(
        (width - (margin_size * 4), height - (margin_size * 4))
    )
    text_subsurface.fill(pygame.Color("black"))

    header = "Current Emotional State:"
    font = pygame.font.SysFont(font_name, font_size)

    header_surface = font.render(header, True, pygame.Color("white"))
    text_subsurface.blit(header_surface, (margin_size, margin_size))

    y_pos = font.size(header)[1] * 2
    total_height = y_pos

    words = emotional_state.split()
    line = ""

    for word in words:
        previous_line = line
        line += word + " "

        line_width = font.size(line)[0]
        line_height = font.size(line)[1]

        if line_width > width - (margin_size * 8):
            line_surface = font.render(previous_line, True, pygame.Color("white"))
            text_subsurface.blit(line_surface, (margin_size, y_pos))
            line = word + " "
            y_pos += line_height
            total_height += line_height

    if line.strip():
        line_surface = font.render(line, True, pygame.Color("white"))
        text_subsurface.blit(line_surface, (margin_size, y_pos))
        total_height += font.size(line)[1]

    text_surface.blit(
        text_subsurface,
        (margin_size, (text_subsurface.get_height() - total_height) // 2),
    )

    surface.blit(text_surface, (margin_size, margin_size))

    return surface


def get_debug_log_surface(
    log_file: str,
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    margin_size: int = 10,
    line_spacing: int = 2,
) -> pygame.Surface:
    surface = pygame.Surface((width, height))
    surface.fill(pygame.Color("black"))

    font = pygame.font.SysFont(font_name, font_size)
    line_height = font.size("A")[1] + line_spacing

    usable_height = height - (margin_size * 2)
    max_lines = usable_height // line_height

    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
    except OSError:
        all_lines = [f"Could not open {log_file}"]

    lines_to_show = [l.rstrip("\n") for l in all_lines[-max_lines:]]

    y_pos = margin_size
    for line in lines_to_show:
        line_surface = font.render(line, True, pygame.Color("white"))
        surface.blit(line_surface, (margin_size, y_pos))
        y_pos += line_height

    return surface


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
