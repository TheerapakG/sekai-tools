import argparse
from collections import defaultdict
import ctypes
from fractions import Fraction
from io import TextIOWrapper
import bisect
import numpy as np
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, TypeVar
import math
import heapq

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", action="store_true")
parser.add_argument("-a", "--audio", action="store_true")

args = parser.parse_args()

chart_id = "0317" # "0329"
unit = "vs"
sus_path = f"./asset/assets/sekai/assetbundle/resources/startapp/music/music_score/{chart_id}_01/master.txt"
music_path = f"./asset/assets/sekai/assetbundle/resources/ondemand/music/long/{unit}_{chart_id}_01/{unit}_{chart_id}_01; {unit}_{chart_id}_01_SCREEN; {unit}_{chart_id}_01_VR.wav"
music_offset = 7.8653998374938965 # 9.010000228881836

TAPS = [None, "normal", "critical", "hold_ignore", None] # type: list[str | None]
TAPS_0 = [None, None, None, None, "skill"] # type: list[str | None]
TAPS_F = [None, "fever_start", "fever_end", None, None] # type: list[str | None]
DIRECTIONS = [None, "flick_up", "hold_ease_in", "flick_upleft", "flick_upright", "hold_ease_out", "hold_ease_out"] # type: list[str | None]

_T = TypeVar("_T")
_U = TypeVar("_U", str, str|None)
_V = TypeVar("_V", str, str|None)

class SUS:
    def __init__(self):
        self.ticks_per_beat = 0
        self.hispeeds = [] # type: list[tuple[int, int, float]]
        self.barlengths = [] # type: list[tuple[int, int]]
        self.bpms = [] # type: list[tuple[int, int, float]]
        self.skills = [] # type: list[tuple[int, int, str|None]]
        self.fevers = [] # type: list[tuple[int, int, str|None]]
        self.notes = [] # type: list[tuple[int, int, int, int, set[str|None]] | tuple[int, int, list[tuple[int, int, int, int, set[str|None]]]]]
        self.combos = [] # type: list[tuple[int, int, int]]

    @classmethod
    def read(cls, f: TextIOWrapper):
        self = cls()

        measure_base = 0
        bpmdefinitions = {} # type: dict[str, float]
        bpms = []  # type: list[tuple[int, Fraction, float]]
        notechannels = defaultdict(list, []) # type: defaultdict[str, list[tuple[int, Fraction, int, int, set[str]]]]
        skills = [] # type: list[tuple[int, Fraction, str|None]]
        fevers = [] # type: list[tuple[int, Fraction, str|None]]
        tap_notes = [] # type: list[tuple[int, Fraction, int, int, set[str|None]]]
        hold_notes = [] # type: list[tuple[int, Fraction, list[tuple[int, Fraction, int, int, set[str]]]]]
        direction_notes = [] # type: list[tuple[int, Fraction, int, int, set[str|None]]]
        parsed_tap_note = set()

        for line in f:
            if not line.startswith("#"):
                continue

            sh_line = shlex.split(line[1:].replace(":", " ", 1).lower()) # type: list[str]
            header = sh_line[0]
            data = sh_line[1] if len(sh_line) > 1 else ""
            header = header.rstrip(":")

            match header:
                case "request":
                    data = data.split()
                    match data[0]:
                        case "ticks_per_beat":
                            self.ticks_per_beat = int(data[1])
                    continue
                case "measurebs":
                    measure_base = int(data)
                    continue
                case "measurehs":
                    continue

            match header[:3]:
                case "bpm":
                    bpmdefinitions[header[3:]] = float(data)
                    continue
                case "til":
                    for match in re.finditer(r"(\d+)'(\d+):(\d*)(?:\.(\d*))?", data):
                        match = match.groups("0")
                        self.hispeeds.append((measure_base + int(match[0]), int(match[1]), float(f"{match[2]}.{match[3]}")))
                    continue

            match [len(header), header[-2:], header[-3], header[-2], header[-1]]:
                case [5, "02", _, _, _]:
                    self.barlengths.append((measure_base + int(header[:-2]), int(data)*self.ticks_per_beat))
                    continue
                case [5, "08", _, _, _]:
                    length = len(data)//2
                    for i, data_unit in enumerate("".join(t) for t in zip(data[::2], data[1::2])):
                        if data_unit == "00":
                            continue
                        bpms.append((measure_base + int(header[:-2]), Fraction(i, length), bpmdefinitions[data_unit]))
                    continue
                case [5, _, _, "1", l]:
                    length = len(data)//2
                    for i, data_unit in enumerate("".join(t) for t in zip(data[::2], data[1::2])):
                        if data_unit == "00":
                            continue
                        match l:
                            case "0":
                                skills.append((measure_base + int(header[:-2]), Fraction(i, length), TAPS_0[int(data_unit[0])]))
                            case "f":
                                fevers.append((measure_base + int(header[:-2]), Fraction(i, length), TAPS_F[int(data_unit[0])]))
                            case _:
                                note_data = (measure_base + int(header[:-2]), Fraction(i, length), int(l, 36) - 2)
                                if note_data in parsed_tap_note:
                                    print(f"duplicated note {note_data}")
                                    continue
                                parsed_tap_note.add(note_data)
                                tap_notes.append((measure_base + int(header[:-2]), Fraction(i, length), int(l, 36) - 2, int(data_unit[1], 36), {TAPS[int(data_unit[0])]}))
                    continue
                case [6, _, "2"|"3"|"4", l, c]:
                    length = len(data)//2
                    for i, data_unit in enumerate("".join(t) for t in zip(data[::2], data[1::2])):
                        if data_unit == "00":
                            continue
                        match int(data_unit[0]):
                            case 1:
                                data = {"hold_start"}
                            case 2:
                                data = {"hold_end"}
                            case 3:
                                data = {"hold_visible"}
                            case 5:
                                data = {"hold_invisible"}
                            case _:
                                data = set()
                        notechannels[c].append((measure_base + int(header[:-3]), Fraction(i, length), int(l, 36) - 2, int(data_unit[1], 36), data))
                    continue
                case [5, _, _, "5", l]:
                    length = len(data)//2
                    for i, data_unit in enumerate("".join(t) for t in zip(data[::2], data[1::2])):
                        if data_unit == "00":
                            continue
                        direction_notes.append((measure_base + int(header[:-2]), Fraction(i, length), int(l, 36) - 2, int(data_unit[1], 36), {DIRECTIONS[int(data_unit[0])]}))
                    continue

            print(f"unrecognized header {header}")

        for channel in notechannels.values():
            channel.sort(key=lambda e: e[:-1])
            current = []
            for n in channel:
                current.append(n)
                if "hold_end" in n[-1]:
                    hold_notes.append((current[0][0],current[0][1], current))
                    current = []
        
        self.hispeeds.sort()
        self.barlengths.sort()

        def bar_fraction_to_ticks(bar, fraction):
            return int(self.barlengths[bisect.bisect_left(self.barlengths, (bar + 1, -1))-1][1] * fraction)

        def map_note(bar: int, fraction: Fraction, *data):
            return bar, bar_fraction_to_ticks(bar, fraction), *data
        
        def map_hold_note(note: tuple[int, Fraction, list[tuple]]):
            bar, fraction, data = note
            return bar, bar_fraction_to_ticks(bar, fraction), tuple(map_note(*n) for n in data)
        
        def combine_note(note: tuple[int, _T, int, int, set[_U]], src_list: list[tuple[int, _T, int, int, set[_V]]]) -> tuple[int, _T, int, int, set[_U|_V]]:
            bar, fraction, l, s, data = note
            left = bisect.bisect_left(src_list, (bar, fraction, l, s), key=lambda e: e[:-1])
            right = bisect.bisect_right(src_list, (bar, fraction, l, s), key=lambda e: e[:-1])
            found = src_list[left:right]
            del src_list[left:right]
            return bar, fraction, l, s, data.union(*(s[-1] for s in found))
        
        self.bpms = sorted(map(lambda e: map_note(*e), bpms))
        self.skills = sorted(map(lambda e: map_note(*e), skills))
        self.fevers = sorted(map(lambda e: map_note(*e), fevers))
        
        tap_notes = sorted(tap_notes)
        direction_notes = sorted(direction_notes)
        tap_direction_notes = sorted([*map(lambda note: combine_note(note, direction_notes), tap_notes), *direction_notes])
        
        tap_direction_notes_m = sorted(map(lambda e: map_note(*e), tap_direction_notes)) # type: list[tuple[int, int, int, int, set[str|None]]]
        hold_notes_m = sorted(map(map_hold_note, hold_notes)) # type: list[tuple[int, int, tuple[tuple[int, int, int, int, set[str]], ...]]]
        
        combined_notes = sorted([*map(lambda hold: (hold[0], hold[1], [combine_note(n, tap_direction_notes_m) for n in hold[2]]), hold_notes_m), *tap_direction_notes_m], key=lambda e: e[:2])

        for note in combined_notes:
            if len(note) == 3:
                if not note[2][0][-1].intersection({"normal", "critical"}):
                    note[2][0][-1].add("normal")
                previous = [*note[2][0][-1].intersection({"normal", "critical"})][0]
                for i in range(1, len(note[2])):
                    if "critical" in note[2][i][-1]:
                        previous = [*{"normal", "critical"}.difference({previous})][0]
                    note[2][i][-1].difference_update({"normal", "critical"})
                    note[2][i][-1].add(previous)

        self.notes = combined_notes

        for bar, fraction, l, s, data in tap_direction_notes_m:
            if "hold_ignore" in data or "hold_ease_in" in data or "hold_ease_out" in data:
                print(bar, fraction, l, s, data)

        events = [] # type: list[tuple[int, int]]
        def add_event(note):
            if "hold_invisible" not in note[-1]:
                events.append(note[0:2])

        for note in self.notes:
            if len(note) != 3:
                add_event(note)
                continue
            for subnote in note[2]:
                add_event(subnote)
            start = list(note[2][0][0:2])
            start_len = self.barlengths[bisect.bisect_left(self.barlengths, (start[0] + 1, -1))-1][1]
            start[1] += (self.ticks_per_beat // 2) - start[1] % (self.ticks_per_beat // 2)
            if start[1] > start_len:
                start[0] += 1
                start[1] -= start_len
            start = tuple(start)
            end = list(note[2][-1][0:2])
            end_len = self.barlengths[bisect.bisect_left(self.barlengths, (end[0] + 1, -1))-1][1]
            if end[1] % (self.ticks_per_beat // 2):
                end[1] += (self.ticks_per_beat // 2) - end[1] % (self.ticks_per_beat // 2)
            if end[1] > end_len:
                end[0] += 1
                end[1] -= end_len
            end = tuple(end)
            if start > end:
                continue
            for bar in range(start[0], end[0] + 1):
                length = int(self.barlengths[bisect.bisect_left(self.barlengths, (bar + 1, -1))-1][1])
                for halfbeat in range(0, length, self.ticks_per_beat // 2):
                    timing = (bar, halfbeat)
                    if timing < start:
                        continue
                    elif timing >= end:
                        break
                    events.append(timing)
        print(len(events))

        events = sorted(events)
        last_timing = None # type: tuple[int, int]|None
        count = 0
        combos = [] # type: list[tuple[int, int, int]]
        for e in events:
            if last_timing and e != last_timing:
                combos.append((*last_timing, count))
            last_timing = e
            count += 1
        if last_timing:
            combos.append((*last_timing, count))

        self.combos = combos

        return self

with open(sus_path) as f:
    sus = SUS.read(f)

from intervaltree import IntervalTree, Interval

t = IntervalTree()
merged_data = sorted(
    [
        *[(e[:2], "hispeed", e[2]) for e in sus.hispeeds],
        *[((e[0], 0), "barlength", e[1]) for e in sus.barlengths],
        *[(e[:2], "bpm", e[2]) for e in sus.bpms],
        *[(e[:2], "skill", e[2]) for e in sus.skills],
        *[(e[:2], "fever", e[2]) for e in sus.fevers],
        *[(e[:2], "note", e[2:]) for e in sus.notes],
        *[(e[:2], "combo", e[2]) for e in sus.combos],
    ],
    key=lambda e: e[0]
)

elapse = 0
y = 0
speed = 1
barlength = 0 # type: int
bpm = 0 # type: float
ticks = (0, 0) # type: tuple[int, int]
pending_notes = [] # type: list[tuple[tuple[int, int], list[tuple[float, float, tuple[int, int, set[str|None]]]], list[tuple[int, int, int, int, set[str|None]]]]]
y_lookup = [] # type: list[tuple[float, float, float]]
inverse_y_lookup = [] # type: list[tuple[float, float, float]]
combo_lookup = [(-math.inf, 0)] # type: list[tuple[float, int]]

for e in merged_data:
    while pending_notes and pending_notes[0][0] < e[0]:
        n_ticks, parsed_subnotes, subnotes = heapq.heappop(pending_notes)
        if barlength and bpm:
            d_ticks = (n_ticks[0] - ticks[0]) * barlength + (n_ticks[1] - ticks[1])
            d_secs = d_ticks / sus.ticks_per_beat / bpm * 60
            elapse += d_secs
            y += d_secs * speed
            ticks = n_ticks
            parsed_subnotes.append((elapse, y, subnotes.pop(0)[2:]))
            if subnotes: 
                heapq.heappush(pending_notes, (subnotes[0][:2], parsed_subnotes, subnotes))
            else:
                t.addi(parsed_subnotes[0][0], elapse, (parsed_subnotes[0][1], ("note", (parsed_subnotes, ))))

    if barlength and bpm:
        d_ticks = (e[0][0] - ticks[0]) * barlength + (e[0][1] - ticks[1])
        d_secs = d_ticks / sus.ticks_per_beat / bpm * 60
        elapse += d_secs
        y += d_secs * speed
        ticks = e[0]
    
    match e:
        case [_, "hispeed", e_speed]:
            speed = e_speed
            if not y_lookup and elapse > 0:
                y_lookup.append((0, 0, 1))
                inverse_y_lookup.append((0, 0, 1))
            y_lookup.append((elapse, y, speed))
            inverse_y_lookup.append((y, elapse, speed))
        case [_, "barlength", e_barlength]:
            barlength = e_barlength
        case [_, "bpm", e_bpm]:
            bpm = e_bpm
        case [_, "note", *_]:
            match e[2]:
                case [_, _, _]:
                    t.addi(elapse, math.nextafter(elapse, math.inf), (y, e[1:]))
                case _:
                    subnotes = e[2][0].copy()
                    parsed_subnotes = [(elapse, y, subnotes.pop(0)[2:])]
                    heapq.heappush(pending_notes, (subnotes[0][:2], parsed_subnotes, subnotes))
        case [_, "combo", e_combo]:
            if not combo_lookup:
                combo_lookup.append((-math.inf, 0))
            combo_lookup.append((elapse, e_combo))


if not y_lookup:
    y_lookup.append((0, 0, 1))
    inverse_y_lookup.append((0, 0, 1))

while pending_notes:
    n_ticks, parsed_subnotes, subnotes = heapq.heappop(pending_notes)
    if barlength and bpm:
        d_ticks = (n_ticks[0] - ticks[0]) * barlength + (n_ticks[1] - ticks[1])
        d_secs = d_ticks / sus.ticks_per_beat / bpm * 60
        elapse += d_secs
        y += d_secs * speed
        ticks = n_ticks
        parsed_subnotes.append((elapse, y, subnotes.pop(0)[2:]))
        if subnotes: 
            heapq.heappush(pending_notes, (subnotes[0][:2], parsed_subnotes, subnotes))
        else:
            t.addi(parsed_subnotes[0][0], elapse, (parsed_subnotes[0][1], ("note", (parsed_subnotes, ))))

from pyglet.window import Window, key
import pyglet.image
import pyglet.media
import pyglet.resource
import pyglet.sprite
from pyglet.graphics.shader import Shader, ShaderProgram
import pyglet.gl as gl

import ffmpeg

pause = True
lead_time = 1
follow_time = 1
previous_offset_time = -lead_time
offset_time = -lead_time
offset_y = 0
previous_time = 0
current_time = 0

def aspect(img: pyglet.image.Texture, b: tuple[int, int], *, fit=True):
    bx, by = b
    ix, iy = img.width, img.height
    sx, sy = bx / ix, by / iy
    s = min(sx, sy) if fit else max(sx, sy)

    return (int(s * ix), int(s * iy))

window = Window(width=1280, height=720)
fps_display = pyglet.window.FPSDisplay(window=window)
keys = key.KeyStateHandler()
window.push_handlers(keys)

textures = {} # type: dict[str, pyglet.image.Texture]
for p in Path("./textures").iterdir():
    if not p.is_file():
        continue
    textures[str(p.relative_to(Path("./textures")).with_suffix(""))] = pyglet.resource.image(str(p.relative_to(".").as_posix()), border=2)

textures["notes_long_among"].anchor_x = textures["notes_long_among"].width / 2
textures["notes_long_among"].anchor_y = textures["notes_long_among"].height / 2
textures["notes_long_among_crtcl"].anchor_x = textures["notes_long_among_crtcl"].width / 2
textures["notes_long_among_crtcl"].anchor_y = textures["notes_long_among_crtcl"].height / 2

media = {} # type: dict[str, pyglet.media.StaticSource]
media_np = {} # type: dict[str, np.ndarray[Any, np.dtype[np.int16]]]
for p in Path("./media").iterdir():
    if not p.is_file():
        continue
    media[str(p.relative_to(Path("./media")).with_suffix(""))] = pyglet.resource.media(str(p.relative_to(".").as_posix()), streaming=False)
    if args.audio:
        out, _ = (
            ffmpeg
            .input(p)
            .output("-", format="s16le", acodec="pcm_s16le", ac="2", ar="44.1k")
            .overwrite_output()
            .run(capture_stdout=True)
        )
        media_np[str(p.relative_to(Path("./media")).with_suffix(""))] = np.frombuffer(out, np.int16).reshape([-1, 2])

music = pyglet.media.load(music_path, streaming=False)
music_np = np.array([], np.float16)
if args.audio:
    out, _ = (
        ffmpeg
        .input(music_path)
        .output("-", format="s16le", acodec="pcm_s16le", ac="2", ar="44.1k")
        .overwrite_output()
        .run(capture_stdout=True)
    )
    music_np = np.frombuffer(out, np.int16).reshape([-1, 2])

music_player = pyglet.media.Player()
music_player.volume = 0.5
music_player.queue(music)

judge_line_y = 160
end_y = 720
judge_line_imaginary_y = 1 / ((1 / judge_line_y) + (1 / (end_y - 0)) * (-0.95))
end_imaginary_y = 1 / ((1 / end_y) + (1 / end_y - 0) * (-0.95))

y_multiplier = 10.8 * 750
judge_line_offset_s = judge_line_imaginary_y / y_multiplier

note_vertex_source = """#version 150 core
    in vec4 colors;
    in vec3 tex_coords;
    in vec2 scale;
    in vec3 position;
    in vec2 note_position;

    out vec4 vertex_colors;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    float calc_note_position;
    vec4 calc_position = vec4(0.0);
    float skew_multiplier = 0.0;
    mat4 m_transform_1 = mat4(1.0);
    mat4 m_transform_2 = mat4(1.0);

    uniform float judge_line_imaginary_y;
    uniform float end_y;
    uniform float end_imaginary_y;
    uniform float offset_y;

    void main()
    {
        calc_note_position = note_position.y - offset_y;

        m_transform_1[0][0] = scale.x;
        m_transform_1[1][1] = scale.y;
        m_transform_1[3][0] = note_position.x - 1 / window.projection[0][0];
        m_transform_1[3][1] = calc_note_position;

        calc_position = m_transform_1 * vec4(position, 1.0);
        skew_multiplier = judge_line_imaginary_y <= calc_note_position && calc_note_position <= end_imaginary_y ? (1.0 + (-0.95 / (((end_y - 0) / calc_position.y) + 0.95))) : 0.0;

        m_transform_2[0][0] = skew_multiplier;
        m_transform_2[1][1] = skew_multiplier;
        m_transform_2[3][0] = 1 / window.projection[0][0];

        gl_Position = window.projection * window.view * m_transform_2 * calc_position;

        vertex_colors = colors;
        texture_coords = tex_coords;
    }
"""

mid_vertex_source = """#version 150 core
    in vec4 colors;
    in vec3 tex_coords;
    in vec2 scale;
    in vec3 position;
    in vec2 note_position;
    in vec2 last_hold_position;
    in vec2 this_hold_position;
    in float hold_ease_outer_pow;
    in float hold_ease_inner_pow;

    out vec4 vertex_colors;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    float calc_note_position;
    vec4 calc_position = vec4(0.0);
    float x_ratio;
    float skew_multiplier = 0.0;
    mat4 m_transform_1 = mat4(1.0);
    mat4 m_transform_2 = mat4(1.0);

    uniform float judge_line_imaginary_y;
    uniform float end_y;
    uniform float end_imaginary_y;
    uniform float offset_y;

    void main()
    {
        calc_note_position = note_position.y - offset_y;
        x_ratio = pow(1 - pow(1 - (note_position.y - last_hold_position.y) / (this_hold_position.y - last_hold_position.y), hold_ease_inner_pow), hold_ease_outer_pow);

        m_transform_1[0][0] = scale.x;
        m_transform_1[1][1] = scale.y;
        m_transform_1[3][0] = mix(last_hold_position.x, this_hold_position.x, x_ratio) - 1 / window.projection[0][0];
        m_transform_1[3][1] = calc_note_position;

        calc_position = m_transform_1 * vec4(position, 1.0);
        skew_multiplier = judge_line_imaginary_y <= calc_note_position && calc_note_position <= end_imaginary_y ? (1.0 + (-0.95 / (((end_y - 0) / calc_note_position) + 0.95))) : 0.0;

        m_transform_2[0][0] = skew_multiplier;
        m_transform_2[1][1] = skew_multiplier;
        m_transform_2[3][0] = 1 / window.projection[0][0];

        gl_Position = window.projection * window.view * m_transform_2 * calc_position;

        vertex_colors = colors;
        texture_coords = tex_coords;
    }
"""

hold_vertex_source = """#version 150 core
    in vec4 colors;
    in vec3 tex_coords;
    in float ratio_y;
    in vec2 last_hold_position;
    in vec2 this_hold_position;
    in float hold_ease_outer_pow;
    in float hold_ease_inner_pow;

    out vec4 vertex_colors;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    float calc_note_position;
    vec4 calc_position = vec4(0.0);
    float x_ratio;
    float skew_multiplier = 0.0;
    mat4 m_transform = mat4(1.0);

    uniform float judge_line_imaginary_y;
    uniform float end_y;
    uniform float end_imaginary_y;
    uniform float offset_y;

    void main()
    {
        calc_position = vec4(-1 / window.projection[0][0], clamp(mix(last_hold_position.y, this_hold_position.y, ratio_y) - offset_y, judge_line_imaginary_y, end_imaginary_y), 0.0, 1.0);
        calc_position.x += mix(last_hold_position.x, this_hold_position.x, pow(1 - pow(1 - (calc_position.y + offset_y - last_hold_position.y) / (this_hold_position.y - last_hold_position.y), hold_ease_inner_pow), hold_ease_outer_pow));
        
        skew_multiplier = (1.0 + (-0.95 / (((end_y - 0) / calc_position.y) + 0.95)));

        m_transform[0][0] = skew_multiplier;
        m_transform[1][1] = skew_multiplier;
        m_transform[3][0] = 1 / window.projection[0][0];
        
        gl_Position = window.projection * window.view * m_transform * calc_position;

        vertex_colors = colors;
        texture_coords = tex_coords;
    }
"""

flick_vertex_source = """#version 150 core
    in vec4 colors;
    in vec3 tex_coords;
    in vec2 scale;
    in vec3 position;
    in vec2 note_position;
    in float x_offset_multiplier;

    out vec4 vertex_colors;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    float calc_note_position;
    vec4 calc_position = vec4(0.0);
    float skew_multiplier = 0.0;
    mat4 m_transform_1 = mat4(1.0);
    mat4 m_transform_2 = mat4(1.0);

    uniform float judge_line_imaginary_y;
    uniform float end_y;
    uniform float end_imaginary_y;
    uniform float offset_y;
    uniform float flick_offset;
    uniform float max_flick_offset;

    void main()
    {
        calc_note_position = note_position.y - offset_y;

        m_transform_1[0][0] = scale.x;
        m_transform_1[1][1] = scale.y;
        m_transform_1[3][0] = note_position.x + (x_offset_multiplier * flick_offset) - 1 / window.projection[0][0];
        m_transform_1[3][1] = flick_offset;

        calc_position = m_transform_1 * vec4(position, 1.0);
        skew_multiplier = judge_line_imaginary_y <= calc_note_position && calc_note_position <= end_imaginary_y ? (1.0 + (-0.95 / (((end_y - 0) / calc_note_position) + 0.95))) : 0.0;

        m_transform_2[0][0] = skew_multiplier;
        m_transform_2[1][1] = skew_multiplier;
        m_transform_2[3][0] = 1 / window.projection[0][0];
        m_transform_2[3][1] = calc_note_position * skew_multiplier;
        
        gl_Position = window.projection * window.view * m_transform_2 * calc_position;

        vertex_colors = colors * vec4(1, 1, 1, pow(1 - (flick_offset / max_flick_offset), 2.0));
        texture_coords = tex_coords;
    }
"""

fragment_source = """#version 150 core
    in vec4 vertex_colors;
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2D sprite_texture;

    void main()
    {
        final_colors = texture(sprite_texture, texture_coords.xy) * vertex_colors;
    }
"""

fragment_array_source = """#version 150 core
    in vec4 vertex_colors;
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2DArray sprite_texture;

    void main()
    {
        final_colors = texture(sprite_texture, texture_coords) * vertex_colors;
    }
"""

note_vertex_shader = Shader(note_vertex_source, "vertex")
mid_vertex_shader = Shader(mid_vertex_source, "vertex")
hold_vertex_shader = Shader(hold_vertex_source, "vertex")
flick_vertex_shader = Shader(flick_vertex_source, "vertex")
fragment_shader = Shader(fragment_source, "fragment")
note_program = ShaderProgram(note_vertex_shader, fragment_shader)
mid_program = ShaderProgram(mid_vertex_shader, fragment_shader)
hold_program = ShaderProgram(hold_vertex_shader, fragment_shader)
flick_program = ShaderProgram(flick_vertex_shader, fragment_shader)

with note_program:
    note_program["judge_line_imaginary_y"] = judge_line_imaginary_y
    note_program["end_y"] = end_y
    note_program["end_imaginary_y"] = end_imaginary_y

with mid_program:
    mid_program["judge_line_imaginary_y"] = judge_line_imaginary_y
    mid_program["end_y"] = end_y
    mid_program["end_imaginary_y"] = end_imaginary_y

with hold_program:
    hold_program["judge_line_imaginary_y"] = judge_line_imaginary_y
    hold_program["end_y"] = end_y
    hold_program["end_imaginary_y"] = end_imaginary_y

with flick_program:
    flick_program["judge_line_imaginary_y"] = judge_line_imaginary_y
    flick_program["end_y"] = end_y
    flick_program["end_imaginary_y"] = end_imaginary_y
    flick_program["max_flick_offset"] = 128

activated = set()
batch = pyglet.graphics.Batch()
bg_group = pyglet.graphics.Group(0)
judge_line_overlay_group = pyglet.graphics.Group(1)
hold_group = pyglet.graphics.Group(2)
note_group = pyglet.graphics.Group(3)
flick_group = pyglet.graphics.Group(4)
hold_judge_line_group = pyglet.graphics.Group(5)
fg_group = pyglet.graphics.Group(6)

sprite_lists = {
    "bg_sprites": [],
    "judge_line_overlay_sprites": [],
    "hold_sprites": [],
    "note_sprites": [],
    "flick_sprites": [],
    "hold_judge_line_sprites": [],
    "fg_sprites": [],
}

flick_offset = 0

NOTES_TEX = ["notes_normal", "notes_crtcl", "notes_long", "notes_flick"]

note_atlas = pyglet.image.atlas.TextureAtlas(512, 1024)
notes_textures = {
    name: note_atlas.add(textures[name].get_image_data(), 2)
    for name in NOTES_TEX
}

notes_coords = {
    name: (
        tex.get_region(0, 0, 82, tex.height).tex_coords,
        tex.get_region(82, 0, tex.width - 164, tex.height).tex_coords,
        tex.get_region(tex.width - 82, 0, 82, tex.height).tex_coords
    )
    for name, tex in notes_textures.items()
}

notes_coords = {
    name: (
        *l_coords[0:3],
        *mid_coords[0:6],
        *r_coords[3:6],
        *r_coords[6:9],
        *mid_coords[6:12],
        *l_coords[9:12],
    )
    for name, (l_coords, mid_coords, r_coords) in notes_coords.items()
}

HOLD_TEX = ["tex_hold_path", "tex_hold_path_crtcl"]

hold_atlas = pyglet.image.atlas.TextureAtlas(512, 128)
hold_textures = {
    name: hold_atlas.add(textures[name].get_image_data(), 2).get_region(32, 1, 448 - 64, 30)
    for name in HOLD_TEX
}

MID_TEX = ["notes_long_among", "notes_long_among_crtcl"]

mid_textures = {
    name: textures[name]
    for name in MID_TEX
}

for tex in mid_textures.values():
    tex.anchor_x = tex.width / 2
    tex.anchor_y = tex.height / 2

FLICK_ARROWS_TEX = [name for i in range(1, 7) for name in [f"notes_flick_arrow_0{i}", f"notes_flick_arrow_crtcl_0{i}"] ]

flick_arrow_textures = {
    name: textures[name]
    for name in FLICK_ARROWS_TEX
}

for tex in flick_arrow_textures.values():
    tex.anchor_x = tex.width / 2

FLICK_ARROW_DIAGONALS_TEX = [name for i in range(1, 7) for name in [f"notes_flick_arrow_0{i}_diagonal", f"notes_flick_arrow_crtcl_0{i}_diagonal"] ]

flick_arrow_diagonal_textures = {}

for i in range(1, 7):
    for name in [f"notes_flick_arrow_0{i}_diagonal", f"notes_flick_arrow_crtcl_0{i}_diagonal"]:
        tex = textures[name]
        tex.anchor_x = tex.width * (2 + (i + 1) / 2) / (2 + (i + 1))
        flick_arrow_diagonal_textures[name] = tex

flick_arrow_textures.update(flick_arrow_diagonal_textures)

def activate_note(data, y):
    if args.video or pause:
        return
    if y > judge_line_imaginary_y + offset_y * y_multiplier:
        return
    if data[:-1] in activated:
        return
    sound = []
    if data[-1].intersection({"flick_up", "flick_upleft", "flick_upright"}):
        sound.append("flick")
    elif "hold_visible" in data[-1]:
        sound.append("connect")

    if "critical" in data[-1]:
        sound.append("critical")

    sound = "se_live_" + ("_".join(sound) if sound else "perfect")

    player = media[sound].play()
    player.volume = 0.5
    activated.add(data[:-1])

note_keys = set()
note_index = defaultdict(lambda: 0)
note_indices = defaultdict(list)
note_position = defaultdict(list)
note_note_position = defaultdict(list)
note_tex_coords = defaultdict(list)
def draw_note_texture(tex_coords, x, y, width = 0, height = 0, key = None):
    if key is None:
        key = "note_sprites"
    note_keys.add(key)
    last_note_index = note_index[key]
    note_index[key] += 8
    note_indices[key].extend(last_note_index + i for i in [0, 1, 6, 0, 6, 7, 1, 2, 5, 1, 5, 6, 2, 3, 4, 2, 4, 5])
    note_position[key].extend((
        -53, -height, 0,
        -53 + 82, -height, 0,
        width + 53 - 82, -height, 0,
        width + 53, -height, 0,

        width + 53, height, 0,
        width + 53 - 82, height, 0,
        -53 + 82, height, 0,
        -53, height, 0,
    ))
    note_note_position[key].extend((x, y) * 8)
    note_tex_coords[key].extend(tex_coords)

def commit_draw_note_texture():
    global note_keys
    global note_index
    global note_indices
    global note_position
    global note_note_position
    global note_tex_coords
    for key in note_keys:
        sprite_lists[key].append(
            note_program.vertex_list_indexed(
                note_index[key],
                pyglet.gl.GL_TRIANGLES,
                note_indices[key],
                batch=batch,
                group=pyglet.sprite.AdvancedSprite.group_class(
                    note_atlas.texture,
                    pyglet.gl.GL_SRC_ALPHA,
                    pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                    note_program, 
                    note_group
                ),
                position=("f", tuple(note_position[key])),
                colors=("Bn", (255, 255, 255, 255) * note_index[key]),
                scale=("f", (1.0, 0.5) * note_index[key]),
                note_position=("f", tuple(note_note_position[key])),
                tex_coords=("f", tuple(note_tex_coords[key]))
            )
        )
    note_keys = set()
    note_index = defaultdict(lambda: 0)
    note_indices = defaultdict(list)
    note_position = defaultdict(list)
    note_note_position = defaultdict(list)
    note_tex_coords = defaultdict(list)

def draw_flick_texture(texture, x, y, reverse, x_offset_mul, sprite_list = None):
    if sprite_list is None:
        sprite_list = sprite_lists["flick_sprites"]
    sprite_list.append(
        flick_program.vertex_list_indexed(
            4,
            pyglet.gl.GL_TRIANGLES,
            [0, 1, 2, 0, 2, 3],
            batch=batch,
            group=pyglet.sprite.AdvancedSprite.group_class(
                texture,
                pyglet.gl.GL_SRC_ALPHA,
                pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                flick_program, 
                flick_group
            ),
            position=(
                "f", 
                (
                    -texture.anchor_x, 0, 0,
                    texture.width - texture.anchor_x, 0, 0,
                    texture.width - texture.anchor_x, texture.height, 0,
                    -texture.anchor_x, texture.height, 0,
                )
            ),
            colors=("Bn", (255, 255, 255, 255) * 4),
            scale=("f", (-1.0 if reverse else 1.0, 1.0) * 4),
            note_position=("f", (x, y) * 4),
            x_offset_multiplier=("f", (x_offset_mul,) * 4),
            tex_coords=("f", texture.tex_coords)
        )
    )

def draw_mid_texture(texture, y, last_hold_position, this_hold_position, ease):
    width = texture.width // 2
    height = texture.height // 2
    sprite_lists["note_sprites"].append(
        mid_program.vertex_list_indexed(
            4,
            pyglet.gl.GL_TRIANGLES,
            [0, 1, 2, 0, 2, 3],
            batch=batch,
            group=pyglet.sprite.AdvancedSprite.group_class(
                texture,
                pyglet.gl.GL_SRC_ALPHA,
                pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                mid_program, 
                note_group
            ),
            position=(
                "f", 
                (
                    -width, -height, 0,
                    width, -height, 0,
                    width, height, 0,
                    -width, height, 0,
                )
            ),
            colors=("Bn", (255, 255, 255, 255) * 4),
            scale=("f", (0.5, 0.5) * 4),
            note_position=("f", (0, y) * 4),
            last_hold_position=("f", last_hold_position * 4),
            this_hold_position=("f", this_hold_position * 4),
            hold_ease_outer_pow=("f", (2.0 if "hold_ease_in" in ease else 1.0,) * 4),
            hold_ease_inner_pow=("f", (2.0 if "hold_ease_out" in ease else 1.0,) * 4),
            tex_coords=("f", texture.tex_coords)
        )
    )

hold_index = 0
hold_indices = []
hold_ratio_y = []
hold_last_hold_position = []
hold_this_hold_position = []
hold_hold_ease_outer_pow = []
hold_hold_ease_inner_pow = []
hold_tex_coords = []
def draw_hold_texture(hold_texture, last_hold_position, this_hold_position, portion, ease):
    global hold_index
    last_hold_index = hold_index
    hold_index += portion * 4
    hold_indices.extend(last_hold_index + e for i in range(0, portion * 4, 4) for e in (i, i+1, i+2, i, i+2, i+3))
    hold_ratio_y.extend(e for i in range(portion) for e in (i / portion, i / portion, (i + 1) / portion, (i + 1) / portion))
    hold_last_hold_position.extend((
        last_hold_position[0], last_hold_position[2],
        last_hold_position[1], last_hold_position[2],
        last_hold_position[1], last_hold_position[2],
        last_hold_position[0], last_hold_position[2],
    ) * portion)
    hold_this_hold_position.extend((
        this_hold_position[0], this_hold_position[2],
        this_hold_position[1], this_hold_position[2],
        this_hold_position[1], this_hold_position[2],
        this_hold_position[0], this_hold_position[2],
    ) * portion)
    hold_hold_ease_outer_pow.extend((2.0 if "hold_ease_in" in ease else 1.0,) * 4 * portion)
    hold_hold_ease_inner_pow.extend((2.0 if "hold_ease_out" in ease else 1.0,) * 4 * portion)
    hold_tex_coords.extend(hold_texture.tex_coords * portion)

def commit_draw_hold_texture():
    global hold_index
    global hold_indices
    global hold_ratio_y
    global hold_last_hold_position
    global hold_this_hold_position
    global hold_hold_ease_outer_pow
    global hold_hold_ease_inner_pow
    global hold_tex_coords
    sprite_lists["hold_sprites"].append(
        hold_program.vertex_list_indexed(
            hold_index,
            pyglet.gl.GL_TRIANGLES,
            hold_indices,
            batch=batch,
            group=pyglet.sprite.AdvancedSprite.group_class(
                hold_atlas.texture,
                pyglet.gl.GL_SRC_ALPHA,
                pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                hold_program, 
                hold_group
            ),
            ratio_y=("f", tuple(hold_ratio_y)),
            last_hold_position=("f", tuple(hold_last_hold_position)),
            this_hold_position=("f", tuple(hold_this_hold_position)),
            hold_ease_outer_pow=("f", tuple(hold_hold_ease_outer_pow)),
            hold_ease_inner_pow=("f", tuple(hold_hold_ease_inner_pow)),
            colors=("Bn", (255, 255, 255, 255) * hold_index),
            tex_coords=("f", tuple(hold_tex_coords)),
        )
    )
    hold_index = 0
    hold_indices = []
    hold_ratio_y = []
    hold_last_hold_position = []
    hold_this_hold_position = []
    hold_hold_ease_outer_pow = []
    hold_hold_ease_inner_pow = []
    hold_tex_coords = []

def draw_flick(data, y, x = None, width = None, sprite_list = None):
    flick_texture = None
    reverse = None
    x_offset_mul = 0
    if "critical" in data[-1]:
        if "flick_up" in data[-1]:
            flick_texture = flick_arrow_textures[f"notes_flick_arrow_crtcl_0{min(6, data[-2])}"]
            reverse = False
        elif "flick_upleft" in data[-1]:
            flick_texture = flick_arrow_textures[f"notes_flick_arrow_crtcl_0{min(6, data[-2])}_diagonal"]
            reverse = False
            x_offset_mul = -0.5
        elif "flick_upright" in data[-1]:
            flick_texture = flick_arrow_textures[f"notes_flick_arrow_crtcl_0{min(6, data[-2])}_diagonal"]
            reverse = True
            x_offset_mul = 0.5
    else:
        if "flick_up" in data[-1]:
            flick_texture = flick_arrow_textures[f"notes_flick_arrow_0{min(6, data[-2])}"]
            reverse = False
        elif "flick_upleft" in data[-1]:
            flick_texture = flick_arrow_textures[f"notes_flick_arrow_0{min(6, data[-2])}_diagonal"]
            reverse = False
            x_offset_mul = -0.5
        elif "flick_upright" in data[-1]:
            flick_texture = flick_arrow_textures[f"notes_flick_arrow_0{min(6, data[-2])}_diagonal"]
            reverse = True
            x_offset_mul = 0.5

    if x is None:
        x = (1280 / 12) * data[-3]
    if width is None:
        width = (1280 / 12) * data[-2]

    if flick_texture:
        draw_flick_texture(flick_texture, x + width / 2, y, reverse, x_offset_mul, sprite_list)

def draw_note(data, y, x = None, width = None, key = None):
    if "critical" in data[-1]:
        texture = "notes_crtcl"
    elif data[-1].intersection({"flick_up", "flick_upleft", "flick_upright"}):
        texture = "notes_flick"
    elif data[-1].intersection({"hold_start", "hold_end"}):
        texture = "notes_long"
    else :
        texture = "notes_normal"
    if x is None:
        x = (1280 / 12) * data[-3]
    if width is None:
        width = (1280 / 12) * data[-2]
    draw_note_texture(notes_coords[texture], x, y, width, notes_textures[texture].height, key)
    draw_flick(data, y, x, width, sprite_lists[key] if key else None)

def draw_mids(hold_data):
    last_data = hold_data[0]
    mids = []
    for data in hold_data[1:]:
        if "hold_visible" in data[2][-1]:
            mids.append(data)
        if "hold_ignore" in data[2][-1]:
            continue
        if "critical" in last_data[2][-1]:
            mid_texture = textures["notes_long_among_crtcl"]
        else:
            mid_texture = textures["notes_long_among"]
        last_y = last_data[1] * y_multiplier
        this_y = data[1] * y_multiplier
        last_x_l = (1280 / 12) * last_data[2][-3]
        last_x_r = last_x_l + (1280 / 12) * last_data[2][-2]
        this_x_l = (1280 / 12) * data[2][-3]
        this_x_r = this_x_l + (1280 / 12) * data[2][-2]
        while mids:
            mid = mids.pop()
            mid_begin, mid_real_y, mid_data = mid
            mid_data = (mid_begin, *mid_data)
            mid_y = mid_real_y * y_multiplier
            draw_mid_texture(mid_texture, mid_y, ((last_x_l + last_x_r) / 2, last_y), ((this_x_l + this_x_r) /2, this_y), last_data[2][-1])
        last_data = data

def draw_hold(hold_data):
    last_data = hold_data[0]
    for data in hold_data[1:]:
        if "hold_ignore" in data[2][-1]:
            continue
        if "critical" in last_data[2][-1]:
            hold_texture = hold_textures["tex_hold_path_crtcl"]
        else:
            hold_texture = hold_textures["tex_hold_path"]
        last_y = last_data[1] * y_multiplier
        this_y = data[1] * y_multiplier
        last_x_l = (1280 / 12) * last_data[2][-3]
        last_x_r = last_x_l + (1280 / 12) * last_data[2][-2]
        this_x_l = (1280 / 12) * data[2][-3]
        this_x_r = this_x_l + (1280 / 12) * data[2][-2]
        ease_in = "hold_ease_in" in last_data[2][-1]
        ease_out = "hold_ease_out" in last_data[2][-1]
        portion = max(math.ceil((this_y - last_y) / 200), 8) if ease_in or ease_out else 1
        draw_hold_texture(hold_texture, (last_x_l, last_x_r, last_y), (this_x_l, this_x_r, this_y), portion, last_data[2][-1])
        last_data = data

def draw_hold_judge_line(hold_data):
    last_data = hold_data[0]
    for data in hold_data[1:]:
        if "hold_ignore" in data[2][-1]:
            continue
        last_y = last_data[1] * y_multiplier
        this_y = data[1] * y_multiplier
        if last_y < judge_line_imaginary_y + offset_y * y_multiplier < this_y:
            last_clamp_y = judge_line_imaginary_y + offset_y * y_multiplier
            ratio = (last_clamp_y - last_y) / (this_y - last_y)
            last_x_l = (1280 / 12) * last_data[2][-3]
            last_x_r = last_x_l + (1280 / 12) * last_data[2][-2]
            this_x_l = (1280 / 12) * data[2][-3]
            this_x_r = this_x_l + (1280 / 12) * data[2][-2]
            ease_in = "hold_ease_in" in last_data[2][-1]
            ease_out = "hold_ease_out" in last_data[2][-1]
            if ease_in:
                x_ratio = ratio ** 2
            elif ease_out:
                x_ratio = 1 - ((1 - ratio) ** 2)
            else:
                x_ratio = ratio
            x_l = last_x_l + (this_x_l - last_x_l) * x_ratio
            x_r = last_x_r + (this_x_r - last_x_r) * x_ratio
            draw_note(hold_data[0][2], judge_line_imaginary_y + offset_y * y_multiplier + 1, x_l, x_r - x_l, key="hold_judge_line_sprites")
            return
        last_data = data

def draw_bg():
    if not sprite_lists["bg_sprites"]:
        sx, sy = aspect(textures["bg_default"], (1280, 720))
        sprite = pyglet.sprite.Sprite(textures["bg_default"], 640 - sx // 2, 360 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["bg_default"].width
        sprite.scale_y = sy / textures["bg_default"].height
        sprite_lists["bg_sprites"].append(sprite)
        sx, sy = aspect(textures["lane_base"], (1280, 960))
        sprite = pyglet.sprite.Sprite(textures["lane_base"], 640 - sx // 2, 360 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["lane_base"].width
        sprite.scale_y = sy / textures["lane_base"].height
        sprite_lists["bg_sprites"].append(sprite)
        sx, sy = aspect(textures["lane_line"], (1280, 960))
        sprite = pyglet.sprite.Sprite(textures["lane_line"], 640 - sx // 2, 360 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["lane_line"].width
        sprite.scale_y = sy / textures["lane_line"].height
        sprite_lists["bg_sprites"].append(sprite)
        sx, sy = aspect(textures["judge_line"], (1080, 120))
        sprite = pyglet.sprite.Sprite(textures["judge_line"], 640 - sx // 2, 150 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["judge_line"].width
        sprite.scale_y = sy / textures["judge_line"].height
        sprite_lists["bg_sprites"].append(sprite)

audio = np.zeros((int((lead_time + t.end() + follow_time) * 44100), 2), np.float64)
def activate_note_audio(data):
    sound = []
    if data[-1].intersection({"flick_up", "flick_upleft", "flick_upright"}):
        sound.append("flick")
    elif "hold_visible" in data[-1]:
        sound.append("connect")

    if "critical" in data[-1]:
        sound.append("critical")

    sound = "se_live_" + ("_".join(sound) if sound else "perfect")

    audio[int((lead_time + data[0]) * 44100):int((lead_time + data[0]) * 44100) + media_np[sound].shape[0], :] += media_np[sound] * 0.5

def commit_note_audio():
    global audio
    music_trim = music_np[int((music_offset - judge_line_offset_s) * 44100):int((music_offset - judge_line_offset_s) * 44100) + audio.shape[0] - int(lead_time * 44100), :]
    music_max = np.max(music_trim)
    audio[int(lead_time * 44100):int(lead_time * 44100) + music_trim.shape[0], :] += music_trim * 0.5
    clipped_audio = (audio / music_max * np.iinfo(np.int16).max).astype(np.int16)
    process = (
        ffmpeg
        .input("-", format="s16le", ac="2", ar="44.1k")
        .output("./out/audio.wav")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    process.stdin.write(clipped_audio)
    process.stdin.close()
    process.wait()
    
def draw_notes():
    for e in sorted(t[:]):
        begin = e.begin
        end = e.end
        real_y, data = e.data
        match data[0]:
            case "note":
                match len(data[1]):
                    case 1:
                        # hold
                        hold_data = data[1][0]
                        # mids
                        draw_mids(hold_data)
                        # start / end
                        for begin, real_y, data in [hold_data[0], hold_data[-1]]:
                            draw_note((begin, *data), real_y * y_multiplier)
                            if args.audio:
                                activate_note_audio((begin, *data))
                        draw_hold(hold_data)
                    case _:
                        # tap
                        draw_note((begin, *data[1]), real_y * y_multiplier)
                        if args.audio:
                            activate_note_audio((begin, *data[1]))
    commit_draw_note_texture()
    commit_draw_hold_texture()
    if args.audio:
        commit_note_audio()

last_combo = 0
last_combo_time = current_time

def draw_fg():
    global last_combo
    global last_combo_time
    combo = combo_lookup[bisect.bisect_right(combo_lookup, offset_time, key=lambda e: e[0]) - 1][1]
    if combo != last_combo:
        last_combo = combo
        last_combo_time = current_time

    if not sprite_lists["fg_sprites"]:
        sprite_lists["fg_sprites"].append(
            pyglet.text.Label(
                "COMBO",
                font_size= 15,
                bold=True,
                color=(255, 255, 255, 255),
                x=1100,
                y=450,
                anchor_x="center",
                anchor_y="center",
                batch=batch,
                group=fg_group
            )
        )
        sprite_lists["fg_sprites"].append(
            pyglet.text.Label(
                f"{combo}",
                font_size=36 + min(1.0, (current_time - last_combo_time) * 8) * 24,
                bold=True,
                color=(255, 255, 255, 255),
                x=1100,
                y=400,
                anchor_x="center",
                anchor_y="center",
                batch=batch,
                group=fg_group
            )
        )
    else:
        if combo:
            sprite_lists["fg_sprites"][1].text = f"{combo}"
            sprite_lists["fg_sprites"][1].font_size = 36 + min(1.0, (current_time - last_combo_time) * 8) * 24
            sprite_lists["fg_sprites"][0].visible = True
            sprite_lists["fg_sprites"][1].visible = True
        else:
            sprite_lists["fg_sprites"][0].visible = False
            sprite_lists["fg_sprites"][1].visible = False

draw_bg()
draw_notes()
draw_fg()

for k, sprite_list in sprite_lists.items():
    print(f"{k}: {len(sprite_list)}")

def update(dt):
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    window.clear()

    global previous_offset_time
    global offset_time
    global previous_time
    global current_time
    global flick_offset

    previous_time = current_time
    current_time = current_time + dt
    if args.video:
        previous_offset_time = offset_time
        offset_time += current_time - previous_time
    elif pause:
        if keys[key.DOWN]:
            offset_time = max(-1, offset_time - 0.05)
        elif keys[key.UP]:
            offset_time += 0.05
        previous_offset_time = math.nextafter(offset_time, -math.inf)
    else:
        previous_offset_time = offset_time
        offset_time += current_time - previous_time

    flick_offset += (current_time - previous_time) * 256
    if flick_offset >= 128:
        flick_offset -= 128

    global music_player
    if not music_player.playing and music_offset + offset_time + judge_line_offset_s > 0 and not pause:
        music_player.seek(music_offset + offset_time + judge_line_offset_s)
        music_player.play()

    global offset_y
    if offset_time < 0:
        offset_y = offset_time
        speed = 1
        offset_end_y = offset_y + end_imaginary_y / y_multiplier
    else:
        base_time, offset_y, speed = y_lookup[bisect.bisect_right(y_lookup, offset_time, key=lambda e: e[0]) - 1]
        offset_y += (offset_time - base_time) * speed
        offset_end_y = offset_y + end_imaginary_y / y_multiplier
    if offset_end_y < 0:
        offset_end_time = offset_end_y
    else:
        base_offset_end_y, offset_end_time, end_speed = inverse_y_lookup[bisect.bisect_right(inverse_y_lookup, offset_end_y, key=lambda e: e[0]) - 1]
        offset_end_time += (offset_end_y - base_offset_end_y) / end_speed
    
    for e in sprite_lists["hold_judge_line_sprites"]:
        e.delete()
        del e
    sprite_lists["hold_judge_line_sprites"] = []

    draw_bg()

    for e in sorted(t[previous_offset_time:offset_time]):
        begin = e.begin
        end = e.end
        real_y, data = e.data
        match data[0]:
            case "note":
                match len(data[1]):
                    case 1:
                        # hold
                        hold_data = data[1][0]
                        # start / end / mids
                        for begin, real_y, data in hold_data:
                            if "hold_invisible" not in data[-1]:
                                activate_note((begin, *data), real_y * y_multiplier)
                        # hold judge line
                        draw_hold_judge_line(hold_data)
                    case _:
                        activate_note((begin, *data[1]), real_y * y_multiplier)    

    commit_draw_note_texture()
    draw_fg()

    activated.add(True)

if args.video:
    process = (
        ffmpeg
        .input('-', format='rawvideo', pixel_format='bgra', s='{}x{}'.format(1280, 720), framerate=60)
        .vflip()
        .output("./out/video.mkv", pix_fmt="yuv420p", vcodec="libx264", tune="animation", preset="veryslow")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    pbo_ids = (gl.GLuint * 2)()
    gl.glGenBuffers(2, pbo_ids)
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo_ids[0])
    gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, 4 * 1280 * 720, 0, gl.GL_DYNAMIC_READ)
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo_ids[1])
    gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, 4 * 1280 * 720, 0, gl.GL_DYNAMIC_READ)
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
    current_pbo = 0
else:
    pyglet.clock.schedule_interval(update, 1 / 60)

@window.event
def on_draw():
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    window.clear()

    if args.video:
        update(1/60)

    with note_program:
        note_program["offset_y"] = offset_y * y_multiplier
    
    with mid_program:
        mid_program["offset_y"] = offset_y * y_multiplier

    with hold_program:
        hold_program["offset_y"] = offset_y * y_multiplier

    with flick_program:
        flick_program["offset_y"] = offset_y * y_multiplier
        flick_program["flick_offset"] = flick_offset

    batch.draw()
    fps_display.draw()

    if args.video:
        global current_pbo
        gl.glReadBuffer(gl.GL_BACK)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo_ids[current_pbo])
        gl.glReadPixels(0, 0, 1280, 720, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, 0)
        current_pbo = (current_pbo + 1) % 2
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo_ids[current_pbo])
        buffer = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)
        process.stdin.write(ctypes.cast(buffer, ctypes.POINTER(gl.GLubyte * (4 * 1280 * 720)))[0])
        gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        if offset_time > t.end() + follow_time:
            gl.glDeleteBuffers(2, pbo_ids)
            process.stdin.close()
            process.wait()
            pyglet.app.exit()

@window.push_handlers
def on_key_press(symbol, modifiers):
    global pause
    global music_player
    global activated
    if not args.video and not args.audio and symbol == key.SPACE:
        pause = not pause
        if not pause:
            if music_offset + offset_time + judge_line_offset_s > 0:
                music_player.seek(music_offset + offset_time + judge_line_offset_s)
                music_player.play()
        else:
            activated = set()
            music_player.pause()

@window.event
def on_resize(width, height):
    pass

pyglet.app.run()

if args.video and args.audio:
    process = (
        ffmpeg
        .input("./out/video.mkv")
        .output(ffmpeg.input("./out/audio.wav"), "./out/mix.mp4", vcodec="copy", acodec="aac", movflags="faststart")
        .overwrite_output()
        .run_async()
    )
    process.wait()
