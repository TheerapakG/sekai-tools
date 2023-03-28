from collections import defaultdict
from fractions import Fraction
from io import TextIOWrapper
import bisect
from pathlib import Path
import re
import shlex
from typing import TypeVar
import math
import heapq

chart_id = "0329"
unit = "vs"
sus_path = f"./asset/assets/sekai/assetbundle/resources/startapp/music/music_score/{chart_id}_01/master.txt"
music_path = f"./asset/assets/sekai/assetbundle/resources/ondemand/music/long/{unit}_{chart_id}_01/{unit}_{chart_id}_01; {unit}_{chart_id}_01_SCREEN; {unit}_{chart_id}_01_VR.wav"
music_offset = 9.010000228881836

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

import time

pause = True
offset_time = -1
offset_y = 0
previous_time = time.perf_counter()
current_time = time.perf_counter()

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
for p in Path("./media").iterdir():
    if not p.is_file():
        continue
    media[str(p.relative_to(Path("./media")).with_suffix(""))] = pyglet.resource.media(str(p.relative_to(".").as_posix()), streaming=False)

music = pyglet.media.load(music_path, streaming=False)
music_player = pyglet.media.Player()
music_player.volume = 0.5
music_player.queue(music)

judge_line_y = 150
end_y = 720
judge_line_imaginary_y = 1 / ((1 / judge_line_y) + (1 / (end_y - 0)) * (-0.95))
end_imaginary_y = 1 / ((1 / end_y) + (1 / end_y - 0) * (-0.95))

y_multiplier = 10.8 * 1000
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
    mat4 m_scale = mat4(1.0);
    mat4 m_translate = mat4(1.0);
    mat4 m_skew = mat4(1.0);

    uniform float judge_line_imaginary_y;
    uniform float end_y;
    uniform float end_imaginary_y;
    uniform float offset_y;

    void main()
    {
        calc_note_position = note_position.y - offset_y;

        m_scale[0][0] = scale.x;
        m_scale[1][1] = scale.y;
        m_translate[3][0] = note_position.x - 640;
        m_translate[3][1] = calc_note_position;

        calc_position = m_translate * m_scale * vec4(position, 1.0);

        skew_multiplier = judge_line_imaginary_y <= calc_note_position && calc_note_position <= end_imaginary_y ? (1.0 + (-0.95 / (((end_y - 0) / calc_position.y) + 0.95))) : 0.0;

        m_skew[0][0] = skew_multiplier;
        m_skew[1][1] = skew_multiplier;
        calc_position = m_skew * calc_position;
        calc_position.x += 640;
        gl_Position = window.projection * window.view * calc_position;

        vertex_colors = colors;
        texture_coords = tex_coords;
    }
"""

hold_vertex_source = """#version 150 core
    in vec4 colors;
    in vec3 tex_coords;
    in vec3 position;

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
    mat4 m_skew = mat4(1.0);

    uniform float end_y;
    uniform float offset_y;

    void main()
    {
        calc_position = vec4(position, 1.0);
        calc_position -= vec4(640, offset_y, 0, 0);

        skew_multiplier = (1.0 + (-0.95 / (((end_y - 0) / calc_position[1]) + 0.95)));
        m_skew[0][0] = skew_multiplier;
        m_skew[1][1] = skew_multiplier;
        calc_position = m_skew * calc_position;
        calc_position += vec4(640, 0, 0, 0);
        gl_Position = window.projection * window.view * calc_position;

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
    mat4 m_scale = mat4(1.0);
    mat4 m_translate = mat4(1.0);
    mat4 m_skew = mat4(1.0);

    uniform float judge_line_imaginary_y;
    uniform float end_y;
    uniform float end_imaginary_y;
    uniform float offset_y;
    uniform float flick_offset;
    uniform float max_flick_offset;

    void main()
    {
        calc_note_position = note_position.y - offset_y;

        m_scale[0][0] = scale.x;
        m_scale[1][1] = scale.y;
        m_translate[3][0] = note_position.x + (x_offset_multiplier * flick_offset) - 640;
        m_translate[3][1] = flick_offset;

        calc_position = m_translate * m_scale * vec4(position, 1.0);

        skew_multiplier = judge_line_imaginary_y <= calc_note_position && calc_note_position <= end_imaginary_y ? (1.0 + (-0.95 / (((end_y - 0) / calc_note_position) + 0.95))) : 0.0;

        m_skew[0][0] = skew_multiplier;
        m_skew[1][1] = skew_multiplier;
        calc_position = m_skew * calc_position;
        calc_position.x += 640;
        calc_position.y += calc_note_position * skew_multiplier;
        gl_Position = window.projection * window.view * calc_position;

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
hold_vertex_shader = Shader(hold_vertex_source, "vertex")
flick_vertex_shader = Shader(flick_vertex_source, "vertex")
fragment_shader = Shader(fragment_source, "fragment")
note_program = ShaderProgram(note_vertex_shader, fragment_shader)
hold_program = ShaderProgram(hold_vertex_shader, fragment_shader)
flick_program = ShaderProgram(flick_vertex_shader, fragment_shader)

with note_program:
    note_program["judge_line_imaginary_y"] = judge_line_imaginary_y
    note_program["end_y"] = end_y
    note_program["end_imaginary_y"] = end_imaginary_y

with hold_program:
    # hold_program["judge_line_imaginary_y"] = judge_line_imaginary_y
    hold_program["end_y"] = end_y
    # hold_program["end_imaginary_y"] = end_imaginary_y

with flick_program:
    flick_program["judge_line_imaginary_y"] = judge_line_imaginary_y
    flick_program["end_y"] = end_y
    flick_program["end_imaginary_y"] = end_imaginary_y
    flick_program["max_flick_offset"] = 128

activated = set()
batch = pyglet.graphics.Batch()
bg_sprites = []
bg_group = pyglet.graphics.Group(0)
hold_sprites = []
hold_group = pyglet.graphics.Group(1)
note_sprites = []
note_group = pyglet.graphics.Group(2)
flick_sprites = []
flick_group = pyglet.graphics.Group(3)
fg_sprites = []
fg_group = pyglet.graphics.Group(4)

flick_offset = 0

NOTES_TEX = ["notes_normal", "notes_crtcl", "notes_long", "notes_flick"]

notes_textures = {
    name: textures[name]
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

hold_textures = {
    name: textures[name].get_region(32, 1, 448 - 64, 30)
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
    if pause:
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

def draw_note_texture(texture, tex_coords, x, y, width = 0, height = 0, sprite_list = None):
    if sprite_list is None:
        sprite_list = note_sprites
    sprite_list.append(
        note_program.vertex_list_indexed(
            12,
            pyglet.gl.GL_TRIANGLES,
            [0, 1, 6, 0, 6, 7, 1, 2, 5, 1, 5, 6, 2, 3, 4, 2, 4, 5],
            batch=batch,
            group=pyglet.sprite.AdvancedSprite.group_class(
                texture,
                pyglet.gl.GL_SRC_ALPHA,
                pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                note_program, 
                note_group
            ),
            position=(
                "f", 
                (
                    -53, -height, 0,
                    -53 + 82, -height, 0,
                    width + 53 - 82, -height, 0,
                    width + 53, -height, 0,

                    width + 53, height, 0,
                    width + 53 - 82, height, 0,
                    -53 + 82, height, 0,
                    -53, height, 0,
                )
            ),
            colors=("Bn", (255, 255, 255, 255) * 8),
            scale=("f", (1.0, 0.5) * 8),
            note_position=("f", (x, y) * 8),
            tex_coords=("f", tex_coords)
        )
    )

def draw_mid_texture(texture, x, y):
    width = texture.width // 2
    height = texture.height // 2
    hold_sprites.append(
        note_program.vertex_list_indexed(
            12,
            pyglet.gl.GL_TRIANGLES,
            [0, 1, 2, 0, 2, 3],
            batch=batch,
            group=pyglet.sprite.AdvancedSprite.group_class(
                texture,
                pyglet.gl.GL_SRC_ALPHA,
                pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                note_program, 
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
            note_position=("f", (x, y) * 4),
            tex_coords=("f", texture.tex_coords)
        )
    )

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
    if sprite_list is None:
        sprite_list = flick_sprites

    if flick_texture:
        flick_width = flick_texture.width
        flick_height = flick_texture.height
        sprite_list.append(
            flick_program.vertex_list_indexed(
                4,
                pyglet.gl.GL_TRIANGLES,
                [0, 1, 2, 0, 2, 3],
                batch=batch,
                group=pyglet.sprite.AdvancedSprite.group_class(
                    flick_texture,
                    pyglet.gl.GL_SRC_ALPHA,
                    pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                    flick_program, 
                    flick_group
                ),
                position=(
                    "f", 
                    (
                        -flick_texture.anchor_x, 0, 0,
                        flick_width - flick_texture.anchor_x, 0, 0,
                        flick_width - flick_texture.anchor_x, flick_height, 0,
                        -flick_texture.anchor_x, flick_height, 0,
                    )
                ),
                colors=("Bn", (255, 255, 255, 255) * 4),
                scale=("f", (-1.0 if reverse else 1.0, 1.0) * 4),
                note_position=("f", (x + width / 2, y) * 4),
                x_offset_multiplier=("f", (x_offset_mul,) * 4),
                tex_coords=("f", flick_texture.tex_coords)
            )
        )

def draw_note(data, y, x = None, width = None, sprite_list = None):
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
    draw_note_texture(notes_textures[texture], notes_coords[texture], x, y, width, notes_textures[texture].height, sprite_list)
    draw_flick(data, y, x, width, sprite_list)

def draw_hold_texture(hold_texture, indices, positions):
    count = len(positions) // 3
    hold_sprites.append(
        hold_program.vertex_list_indexed(
            count,
            pyglet.gl.GL_TRIANGLES,
            indices,
            batch=batch,
            group=pyglet.sprite.AdvancedSprite.group_class(
                hold_texture,
                pyglet.gl.GL_SRC_ALPHA,
                pyglet.gl.GL_ONE_MINUS_SRC_ALPHA,
                hold_program, 
                hold_group
            ),
            position=(
                "f", 
                tuple(positions)
            ),
            colors=("Bn", (255, 255, 255, 255) * count),
            tex_coords=("f", hold_texture.tex_coords * (count // 4))
        )
    )

def draw_hold(hold_data, y):
    last_data = hold_data[0]
    mids = []
    index = 0
    indices = []
    positions = []
    hold_texture_str = None
    hold_texture = None
    for data in hold_data[1:]:
        if "hold_visible" in data[2][-1]:
            mids.append(data)
        if "hold_ignore" in data[2][-1]:
            continue
        if "critical" in last_data[2][-1]:
            new_hold_texture_str = "tex_hold_path_crtcl"
            mid_texture = "notes_long_among_crtcl"
        else:
            new_hold_texture_str = "tex_hold_path"
            mid_texture = "notes_long_among"
        if indices and positions and (new_hold_texture_str != hold_texture_str):
            draw_hold_texture(hold_texture, indices, positions)
            index = 0
            indices = []
            positions = []
        hold_texture_str = new_hold_texture_str
        hold_texture = hold_textures[new_hold_texture_str]
        mid_texture = textures[mid_texture]
        last_y = (last_data[1] - hold_data[0][1]) * y_multiplier + y
        this_y = (data[1] - hold_data[0][1]) * y_multiplier + y
        if this_y - offset_y * y_multiplier < judge_line_imaginary_y:
            last_data = data
            continue
        elif last_y - offset_y * y_multiplier > end_imaginary_y:
            break
        last_x_l = (1280 / 12) * last_data[2][-3]
        last_x_r = last_x_l + (1280 / 12) * last_data[2][-2]
        this_x_l = (1280 / 12) * data[2][-3]
        this_x_r = this_x_l + (1280 / 12) * data[2][-2]
        ease_in = "hold_ease_in" in last_data[2][-1]
        ease_out = "hold_ease_out" in last_data[2][-1]
        if last_y < judge_line_imaginary_y + offset_y * y_multiplier:
            y_ratio = (judge_line_imaginary_y + offset_y * y_multiplier - last_y) / (this_y - last_y)
            if ease_in:
                y_ratio = y_ratio ** 2
            elif ease_out:
                y_ratio = 1 - ((1 - y_ratio) ** 2)
            last_x_l = (this_x_l - last_x_l) * y_ratio + last_x_l
            last_x_r = (this_x_r - last_x_r) * y_ratio + last_x_r
            last_y = judge_line_imaginary_y + offset_y * y_multiplier
        if this_y > end_imaginary_y + offset_y * y_multiplier:
            y_ratio = (end_imaginary_y + offset_y * y_multiplier - last_y) / (this_y - last_y)
            if ease_in:
                y_ratio = y_ratio ** 2
            elif ease_out:
                y_ratio = 1 - ((1 - y_ratio) ** 2)
            this_x_l = (this_x_l - last_x_l) * y_ratio + last_x_l
            this_x_r = (this_x_r - last_x_r) * y_ratio + last_x_r
            this_y = end_imaginary_y + offset_y * y_multiplier
        last_real_y = (1 + (-0.95 / ((600 / (last_y - offset_y * y_multiplier)) + 0.95))) * (last_y - offset_y * y_multiplier)
        this_real_y = (1 + (-0.95 / ((600 / (this_y - offset_y * y_multiplier)) + 0.95))) * (this_y - offset_y * y_multiplier)
        portion = max(round((this_real_y - last_real_y) / (end_y - judge_line_y) * 32), 8) if ease_in or ease_out else 1
        ratios = [i / portion for i in range(portion + 1)]
        if ease_in:
            x_ratios = [ratio**2 for ratio in ratios]
        elif ease_out:
            x_ratios = [1 - ((1 - ratio) ** 2) for ratio in ratios]
        else:
            x_ratios = ratios
        d_y = this_y - last_y
        portion_y = [d_y * ratio + last_y for ratio in ratios]
        d_x_l = this_x_l - last_x_l
        portion_x_l = [d_x_l * ratio + last_x_l for ratio in x_ratios]
        d_x_r = this_x_r - last_x_r
        portion_x_r = [d_x_r * ratio + last_x_r for ratio in x_ratios]
        portion_data = [*zip(portion_x_l, portion_x_r, portion_y)]
        for last_portion, this_portion in zip(portion_data[:-1], portion_data[1:]):
            last_l_index = index
            index += 1
            positions.extend((last_portion[0], last_portion[2], 0))
            last_r_index = index
            index += 1
            positions.extend((last_portion[1], last_portion[2], 0))
            this_r_index = index
            index += 1
            positions.extend((this_portion[1], this_portion[2], 0))
            this_l_index = index
            index += 1
            positions.extend((this_portion[0], this_portion[2], 0))
            indices.extend((last_l_index, last_r_index, this_r_index, last_l_index, this_r_index, this_l_index))
        while mids:
            mid = mids.pop()
            mid_begin, mid_real_y, mid_data = mid
            mid_data = (mid_begin, *mid_data)
            mid_y = (mid_real_y - hold_data[0][1]) * y_multiplier + y
            activate_note(mid_data, mid_y)
            if mid_y > end_imaginary_y + offset_y * y_multiplier:
                continue
            mid_ratio = (mid_y - last_y) / (this_y - last_y)
            if "hold_ease_in" in last_data[2][-1]:
                mid_ratio = mid_ratio ** 2
            elif "hold_ease_out" in last_data[2][-1]:
                mid_ratio = 1 - ((1 - mid_ratio) ** 2)
            mid_x_l = int(((this_x_l - last_x_l) * mid_ratio) + last_x_l)
            mid_x_r = int(((this_x_r - last_x_r) * mid_ratio) + last_x_r)
            draw_mid_texture(mid_texture, (mid_x_l + mid_x_r) // 2, mid_y)
        if portion_data[0][-1] == judge_line_imaginary_y + offset_y * y_multiplier:
            draw_note(hold_data[0][2], judge_line_imaginary_y + offset_y * y_multiplier + 1, portion_data[0][0], portion_data[0][1] - portion_data[0][0], sprite_list=hold_sprites)
        last_data = data
    if indices and positions:
        draw_hold_texture(hold_texture, indices, positions)

def draw_bg():
    if not bg_sprites:
        sx, sy = aspect(textures["bg_default"], (1280, 720))
        sprite = pyglet.sprite.Sprite(textures["bg_default"], 640 - sx // 2, 360 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["bg_default"].width
        sprite.scale_y = sy / textures["bg_default"].height
        bg_sprites.append(sprite)
        sx, sy = aspect(textures["lane_base"], (1280, 960))
        sprite = pyglet.sprite.Sprite(textures["lane_base"], 640 - sx // 2, 360 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["lane_base"].width
        sprite.scale_y = sy / textures["lane_base"].height
        bg_sprites.append(sprite)
        sx, sy = aspect(textures["lane_line"], (1280, 960))
        sprite = pyglet.sprite.Sprite(textures["lane_line"], 640 - sx // 2, 360 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["lane_line"].width
        sprite.scale_y = sy / textures["lane_line"].height
        bg_sprites.append(sprite)
        sx, sy = aspect(textures["judge_line"], (1080, 120))
        sprite = pyglet.sprite.Sprite(textures["judge_line"], 640 - sx // 2, 150 - sy // 2, batch=batch, group=bg_group)
        sprite.scale_x = sx / textures["judge_line"].width
        sprite.scale_y = sy / textures["judge_line"].height
        bg_sprites.append(sprite)

def draw_notes():
    for e in sorted(t[:]):
        begin = e.begin
        end = e.end
        real_y, data = e.data
        match data[0]:
            case "hispeed":
                pass
            case "barlength":
                pass
            case "bpm":
                pass
            case "skill":
                pass
            case "fever":
                pass
            case "note":
                match len(data[1]):
                    case 1:
                        # hold
                        hold_data = data[1][0]
                        # start / end
                        for begin, real_y, data in [hold_data[0], hold_data[-1]]:
                            draw_note((begin, *data), real_y * y_multiplier)
                    case _:
                        # tap
                        draw_note((begin, *data[1]), real_y * y_multiplier)

last_combo = 0
last_combo_time = current_time

def draw_fg():
    global last_combo
    global last_combo_time
    combo = combo_lookup[bisect.bisect_right(combo_lookup, offset_time, key=lambda e: e[0]) - 1][1]
    if combo != last_combo:
        last_combo = combo
        last_combo_time = current_time

    if not fg_sprites:
        fg_sprites.append(
                pyglet.text.Label(
                f"{len(hold_sprites)}|{len(note_sprites)}|{len(flick_sprites)}",
                font_size=24,
                bold=True,
                color=(127, 127, 127, 127),
                x=10,
                y=48,
                batch=batch,
                group=fg_group
            )
        )
        fg_sprites.append(
            pyglet.text.Label(
                "COMBO",
                font_size= 20,
                bold=True,
                color=(255, 255, 255, 255),
                x=1100,
                y=460,
                anchor_x="center",
                anchor_y="center",
                batch=batch,
                group=fg_group
            )
        )
        fg_sprites.append(
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
        fg_sprites[0].text = f"{len(hold_sprites)}|{len(note_sprites)}|{len(flick_sprites)}"
        if combo:
            fg_sprites[2].text = f"{combo}"
            fg_sprites[2].font_size = 36 + min(1.0, (current_time - last_combo_time) * 8) * 24
            fg_sprites[1].visible = True
            fg_sprites[2].visible = True
        else:
            fg_sprites[1].visible = False
            fg_sprites[2].visible = False

draw_bg()
draw_notes()
draw_fg()

@window.event
def on_draw():
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    window.clear()

    global offset_time
    global previous_time
    global current_time
    global flick_offset
    current_time = time.perf_counter()

    previous_offset_time = offset_time

    if pause:
        if keys[key.DOWN]:
            offset_time = max(-1, offset_time - 0.05)
        elif keys[key.UP]:
            offset_time += 0.05
    else:
        offset_time += current_time - previous_time

    flick_offset += (current_time - previous_time) * 256
    if flick_offset >= 128:
        flick_offset -= 128

    global music_player
    if not music_player.playing and not pause and music_offset + offset_time + judge_line_offset_s > 0:
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

    global bg_sprites
    global hold_sprites
    global note_sprites
    global flick_sprites
    global fg_sprites
    
    for e in hold_sprites:
        e.delete()
        del e
    hold_sprites = []

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
                        # start / end
                        for begin, real_y, data in [hold_data[0], hold_data[-1]]:
                            activate_note((begin, *data), real_y * y_multiplier)
                    case _:
                        activate_note((begin, *data[1]), real_y * y_multiplier)

    for e in sorted(t[offset_time:offset_end_time]):
        begin = e.begin
        end = e.end
        real_y, data = e.data
        match data[0]:
            case "note":
                match len(data[1]):
                    case 1:
                        # hold
                        hold_data = data[1][0]
                        # hold path
                        draw_hold(hold_data, real_y * y_multiplier)

    draw_fg()

    with note_program:
        note_program["offset_y"] = offset_y * y_multiplier

    with hold_program:
        hold_program["offset_y"] = offset_y * y_multiplier

    with flick_program:
        flick_program["offset_y"] = offset_y * y_multiplier
        flick_program["flick_offset"] = flick_offset

    batch.draw()

    activated.add(True)
    previous_time = current_time
    fps_display.draw()

@window.push_handlers
def on_key_press(symbol, modifiers):
    global pause
    global music_player
    global activated
    if symbol == key.SPACE:
        pause = not pause
        if not pause:
            if music_offset + offset_time + judge_line_offset_s > 0:
                music_player.seek(music_offset + offset_time + judge_line_offset_s)
                music_player.play()
        else:
            activated = set()
            music_player.pause()

pyglet.app.run()
