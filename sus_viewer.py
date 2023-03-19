from collections import defaultdict
from fractions import Fraction
from io import TextIOWrapper
import bisect
from pathlib import Path
import re
import shlex
import math
import heapq

sus_path = "./asset/assets/sekai/assetbundle/resources/startapp/music/music_score/0329_01/master.txt"
music_path = "./asset/assets/sekai/assetbundle/resources/ondemand/music/long/vs_0329_01/vs_0329_01; vs_0329_01_SCREEN; vs_0329_01_VR.wav"
music_offset = 9.010000228881836

TAPS = [None, "normal", "critical", "hold_ignore", None]
TAPS_0 = [None, None, None, None, "skill"]
TAPS_F = [None, "fever_start", "fever_end", None, None]
DIRECTIONS = [None, "flick_up", "hold_ease_in", "flick_upleft", "flick_upright", "hold_ease_out", "hold_ease_out"]

class SUS:
    def __init__(self):
        self.ticks_per_beat = 0
        self.hispeeds = []
        self.barlengths = []
        self.bpms = []
        self.skills = []
        self.fevers = []
        self.notes = []

    @classmethod
    def read(cls, f: TextIOWrapper):
        self = cls()

        measure_base = 0
        bpmdefinitions = {}
        bpms = []
        notechannels = defaultdict(list, [])
        skills = []
        fevers = []
        tap_notes = []
        hold_notes = []
        direction_notes = []

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
                    self.barlengths.append((measure_base + int(header[:-2]), float(data)*self.ticks_per_beat))
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

        self.bpms = sorted(map(lambda e: (e[0], bar_fraction_to_ticks(e[0], e[1]), e[2]), bpms))
        self.skills = sorted(map(lambda e: (e[0], bar_fraction_to_ticks(e[0], e[1]), e[2]), skills))
        self.fevers = sorted(map(lambda e: (e[0], bar_fraction_to_ticks(e[0], e[1]), e[2]), fevers))

        def map_note(note):
            bar, fraction, l, s, data = note
            return bar, bar_fraction_to_ticks(bar, fraction), l, s, data
        
        def map_hold_note(note):
            bar, fraction, data = note
            return bar, bar_fraction_to_ticks(bar, fraction), tuple(map_note(n) for n in data)
        
        tap_notes = sorted(map(map_note, tap_notes))
        hold_notes = sorted(map(map_hold_note, hold_notes))
        direction_notes = sorted(map(map_note, direction_notes))

        def combine_note(note, src_list):
            bar, fraction, l, s, data = note
            left = bisect.bisect_left(src_list, (bar, fraction, l, s), key=lambda e: e[:-1])
            right = bisect.bisect_right(src_list, (bar, fraction, l, s), key=lambda e: e[:-1])
            found = src_list[left:right]
            del src_list[left:right]
            return bar, fraction, l, s, data.union(*(s[-1] for s in found))
        
        tap_direction_notes = sorted([*map(lambda note: combine_note(note, direction_notes), tap_notes), *direction_notes])
        combined_notes = sorted([*map(lambda hold: (hold[0], hold[1], [combine_note(n, tap_direction_notes) for n in hold[2]]), hold_notes), *tap_direction_notes], key=lambda e: e[:2])

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

        for bar, fraction, l, s, data in tap_direction_notes:
            if "hold_ignore" in data or "hold_ease_in" in data or "hold_ease_out" in data:
                print(bar, fraction, l, s, data)

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
    ],
    key=lambda e: e[0]
)

elapse = 0
y = 0
speed = 1
barlength = 0
bpm = 0
ticks = (0, 0)
pending_notes = []
y_lookup = []
inverse_y_lookup = []

for e in merged_data:
    while pending_notes and pending_notes[0][0] < e[0]:
        n_ticks, i, subnotes = heapq.heappop(pending_notes)
        if barlength and bpm:
            d_ticks = (n_ticks[0] - ticks[0]) * barlength + (n_ticks[1] - ticks[1])
            d_secs = d_ticks / sus.ticks_per_beat / bpm * 60
            elapse += d_secs
            y += d_secs * speed
            ticks = n_ticks
            subnotes[i] = (elapse, y, subnotes[i][2:])
            if i + 1 < len(subnotes): 
                heapq.heappush(pending_notes, (subnotes[i + 1][:2], i + 1, subnotes))
            else:
                t.addi(subnotes[0][0], elapse, (subnotes[0][1], ("note", (subnotes, ))))

    if barlength and bpm:
        d_ticks = (e[0][0] - ticks[0]) * barlength + (e[0][1] - ticks[1])
        d_secs = d_ticks / sus.ticks_per_beat / bpm * 60
        elapse += d_secs
        y += d_secs * speed
        ticks = e[0]
    
    match e[1]:
        case "hispeed":
            speed = e[2]
            t.addi(elapse, math.nextafter(elapse, math.inf), (y, e[1:]))
            if not y_lookup and elapse > 0:
                y_lookup.append((0, 0, 1))
                inverse_y_lookup.append((0, 0, 1))
            y_lookup.append((elapse, y, speed))
            inverse_y_lookup.append((y, elapse, speed))
        case "barlength":
            barlength = e[2]
            t.addi(elapse, math.nextafter(elapse, math.inf), (y, e[1:]))
        case "bpm":
            bpm = e[2]
            t.addi(elapse, math.nextafter(elapse, math.inf), (y, e[1:]))
        case "skill":
            t.addi(elapse, math.nextafter(elapse, math.inf), (y, e[1:]))
        case "fever":
            t.addi(elapse, math.nextafter(elapse, math.inf), (y, e[1:]))
        case "note":
            if len(e[2]) == 3:
                t.addi(elapse, math.nextafter(elapse, math.inf), (y, e[1:]))
            else:
                subnotes = e[2][0].copy()
                subnotes[0] = (elapse, y, subnotes[0][2:])
                heapq.heappush(pending_notes, (subnotes[1][:2], 1, subnotes))

while pending_notes:
    n_ticks, i, subnotes = heapq.heappop(pending_notes)
    if barlength and bpm:
        d_ticks = (n_ticks[0] - ticks[0]) * barlength + (n_ticks[1] - ticks[1])
        d_secs = d_ticks / sus.ticks_per_beat / bpm * 60
        elapse += d_secs
        y += d_secs * speed
        ticks = n_ticks
        subnotes[i] = (elapse, y, subnotes[i][2:])
        if i + 1 < len(subnotes): 
            heapq.heappush(pending_notes, (subnotes[i + 1][:2], i + 1, subnotes))
        else:
            print(subnotes[0][1], subnotes)
            t.addi(subnotes[0][0], elapse, (subnotes[0][1], ("note", (subnotes, ))))

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
previous_time = time.perf_counter()

def aspect(img: pyglet.image.Texture, b: tuple[int, int], *, fit=True):
    bx, by = b
    ix, iy = img.width, img.height
    sx, sy = bx / ix, by / iy
    s = min(sx, sy) if fit else max(sx, sy)

    return (int(s * ix), int(s * iy))

window = Window(width=1280, height=720)
keys = key.KeyStateHandler()
window.push_handlers(keys)

textures = {} # type: dict[str, pyglet.image.Texture]
for p in Path("./textures").iterdir():
    if not p.is_file():
        continue
    textures[str(p.relative_to(Path("./textures")).with_suffix(""))] = pyglet.resource.image(str(p.relative_to(".")), border=2)

media = {} # type: dict[str, pyglet.media.StaticSource]
for p in Path("./media").iterdir():
    if not p.is_file():
        continue
    media[str(p.relative_to(Path("./media")).with_suffix(""))] = pyglet.resource.media(str(p.relative_to(".")), streaming=False)

music = pyglet.media.load(music_path)
music_player = None

skew_vertex_source = """#version 150 core
    in vec3 translate;
    in vec4 colors;
    in vec3 tex_coords;
    in vec2 scale;
    in vec3 position;
    in float rotation;

    out vec4 vertex_colors;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    vec4 calc_position = vec4(0.0);
    float skew_multiplier = 0.0;
    mat4 m_scale = mat4(1.0);
    mat4 m_rotation = mat4(1.0);
    mat4 m_translate = mat4(1.0);
    mat4 m_skew = mat4(1.0);

    void main()
    {
        m_scale[0][0] = scale.x;
        m_scale[1][1] = scale.y;
        m_translate[3][0] = translate.x;
        m_translate[3][1] = translate.y;
        m_translate[3][2] = translate.z;
        m_rotation[0][0] =  cos(-radians(rotation)); 
        m_rotation[0][1] =  sin(-radians(rotation));
        m_rotation[1][0] = -sin(-radians(rotation));
        m_rotation[1][1] =  cos(-radians(rotation));

        calc_position = m_translate * m_rotation * m_scale * vec4(position, 1.0);
        calc_position -= vec4(640, 0, 0, 0);
        skew_multiplier = (1.0 + (-0.95 / ((600.0 / calc_position[1]) + 0.95)));
        m_skew[0][0] = skew_multiplier;
        m_skew[1][1] = skew_multiplier;
        calc_position = m_skew * calc_position;
        calc_position += vec4(640, 0, 0, 0);
        gl_Position = window.projection * window.view * calc_position;

        vertex_colors = colors;
        texture_coords = tex_coords;
    }
"""

skew_fragment_source = """#version 150 core
    in vec4 vertex_colors;
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2D sprite_texture;

    void main()
    {
        final_colors = texture(sprite_texture, texture_coords.xy) * vertex_colors;
    }
"""

skew_fragment_array_source = """#version 150 core
    in vec4 vertex_colors;
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2DArray sprite_texture;

    void main()
    {
        final_colors = texture(sprite_texture, texture_coords) * vertex_colors;
    }
"""

skew_vertex_shader = Shader(skew_vertex_source, "vertex")
skew_fragment_shader = Shader(skew_fragment_source, "fragment")
skew_program = ShaderProgram(skew_vertex_shader, skew_fragment_shader)
skew_array_program = ShaderProgram(skew_vertex_shader, skew_fragment_shader)

NOTES_TEX = ["notes_normal", "notes_crtcl", "notes_long", "notes_flick"]

notes_textures = {
    tex: pyglet.image.TextureGrid(pyglet.image.ImageGrid(textures[tex], 1, 3))
    for tex in NOTES_TEX
}

judge_line_imaginary_y = 1 / ((1 / 120) + (1 / 600) * (-0.95))
judge_line_offset_s = judge_line_imaginary_y / 10.8 / 400

y_multiplier = 10.8 * 500

activated = set()
sprites = []
batch = pyglet.graphics.Batch()

def activate_note(data):
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

def draw_note(data, y):
    if y < judge_line_imaginary_y:
        if data[:-1] not in activated:
            activate_note(data)
        return
    if "critical" in data[-1]:
        texture = "notes_crtcl"
    elif data[-1].intersection({"flick_up", "flick_upleft", "flick_upright"}):
        texture = "notes_flick"
    elif data[-1].intersection({"hold_start", "hold_end"}):
        texture = "notes_long"
    else :
        texture = "notes_normal"
    x = 125 + ((1280 - 250) / 12) * data[-3]
    sprites.append(pyglet.sprite.AdvancedSprite(notes_textures[texture][0], x - 53, y, batch = batch, program=skew_program))
    middle_sprite = pyglet.sprite.AdvancedSprite(notes_textures[texture][1], x + 65, y, batch = batch, program=skew_program)
    middle_sprite.scale_x = (((1280 - 250) / 12) * data[-2] - 130) / 118
    sprites.append(middle_sprite)
    sprites.append(pyglet.sprite.AdvancedSprite(notes_textures[texture][2], x + ((1280 - 250) / 12) * data[-2] - 65, y, batch = batch, program=skew_program))

    flick_texture = None
    reverse = None
    if texture == "notes_crtcl":
        if "flick_up" in data[-1]:
            flick_texture = f"notes_flick_arrow_crtcl_0{min(6, data[-2])}"
            reverse = False
        elif "flick_upleft" in data[-1]:
            flick_texture = f"notes_flick_arrow_crtcl_0{min(6, data[-2])}_diagonal"
            reverse = False
        elif "flick_upright" in data[-1]:
            flick_texture = f"notes_flick_arrow_crtcl_0{min(6, data[-2])}_diagonal"
            reverse = True
    elif texture == "notes_flick":
        if "flick_up" in data[-1]:
            flick_texture = f"notes_flick_arrow_0{min(6, data[-2])}"
            reverse = False
        elif "flick_upleft" in data[-1]:
            flick_texture = f"notes_flick_arrow_0{min(6, data[-2])}_diagonal"
            reverse = False
        elif "flick_upright" in data[-1]:
            flick_texture = f"notes_flick_arrow_0{min(6, data[-2])}_diagonal"
            reverse = True

    if flick_texture:
        flick_sprite = pyglet.sprite.AdvancedSprite(textures[flick_texture], x + ((((1280 - 250) / 12) * data[-2]) / 2) - (textures[flick_texture].width / 2  * (-1 if reverse else 1)), y + 100, batch=batch, program=skew_program)
        if reverse:
            flick_sprite.scale_x = -1
        sprites.append(flick_sprite)

def draw_hold(hold_data, y):
    last_data = hold_data[0]
    mids = []
    for data in hold_data[1:]:
        if "hold_visible" in data[2][-1]:
            mids.append(data)
        if "hold_ignore" in data[2][-1]:
            continue
        if "critical" in last_data[2][-1]:
            hold_texture = "tex_hold_path_crtcl"
            mid_texture = "notes_long_among_crtcl"
        else:
            hold_texture = "tex_hold_path"
            mid_texture = "notes_long_among"
        hold_texture = textures[hold_texture]
        mid_texture = textures[mid_texture]
        last_y = (last_data[1] - hold_data[0][1]) * y_multiplier + y
        this_y = (data[1] - hold_data[0][1]) * y_multiplier + y
        if last_y + 93 < judge_line_imaginary_y and this_y + 93 < judge_line_imaginary_y or last_y + 93 > 600 / 0.05 / 2.5 and this_y + 93 > 600 / 0.05 / 2.5:
            last_data = data
            continue
        last_x_l = 125 + ((1280 - 250) / 12) * last_data[2][-3]
        last_x_r = last_x_l + ((1280 - 250) / 12) * last_data[2][-2]
        this_x_l = 125 + ((1280 - 250) / 12) * data[2][-3]
        this_x_r = this_x_l + ((1280 - 250) / 12) * data[2][-2]
        if last_y + 93 < judge_line_imaginary_y:
            y_ratio = (judge_line_imaginary_y - last_y - 93) / (this_y - last_y)
            if "hold_ease_in" in last_data[2][-1]:
                y_ratio = y_ratio ** 2
            elif "hold_ease_out" in last_data[2][-1]:
                y_ratio = 1 - ((1 - y_ratio) ** 2)
            last_x_l = (this_x_l - last_x_l) * y_ratio + last_x_l
            last_x_r = (this_x_r - last_x_r) * y_ratio + last_x_r
            last_y = judge_line_imaginary_y - 93
        if this_y + 93 > 600 / 0.05 / 2.5:
            y_ratio = (600 / 0.05 / 2.5 - last_y - 93) / (this_y - last_y)
            if "hold_ease_in" in last_data[2][-1]:
                y_ratio = y_ratio ** 2
            elif "hold_ease_out" in last_data[2][-1]:
                y_ratio = 1 - ((1 - y_ratio) ** 2)
            this_x_l = (this_x_l - last_x_l) * y_ratio + last_x_l
            this_x_r = (this_x_r - last_x_r) * y_ratio + last_x_r
            this_y = 600 / 0.05 / 2.5 - 93
        min_portion = 8 if last_data[2][-1].intersection({"hold_ease_in", "hold_ease_out"}) else 1
        portion = max(round((this_y - last_y) / (600 / 0.05 / 2.5 - judge_line_imaginary_y) * 128), min_portion)
        ratios = [i / portion for i in range(portion + 1)]
        if "hold_ease_in" in last_data[2][-1]:
            x_ratios = [ratio**2 for ratio in ratios]
        elif "hold_ease_out" in last_data[2][-1]:
            x_ratios = [1 - ((1 - ratio) ** 2) for ratio in ratios]
        else:
            x_ratios = ratios
        d_y = this_y - last_y
        portion_y = [d_y * ratio + last_y for ratio in ratios]
        d_x_l = this_x_l - last_x_l
        portion_x_l = [d_x_l * ratio + last_x_l for ratio in x_ratios]
        d_x_r = this_x_r - last_x_r
        portion_x_r = [d_x_r * ratio + last_x_r for ratio in x_ratios]
        portion_width = [x_r - x_l for x_l, x_r in zip(portion_x_l, portion_x_r)]
        portion_x_l = [int(x_l - (width // 12))for x_l, width in zip(portion_x_l, portion_width)]
        portion_x_r = [int(x_r + (width // 12))for x_r, width in zip(portion_x_r, portion_width)]
        portion_data = [*zip(portion_x_l, portion_x_r, portion_y)]
        for last_portion, this_portion in zip(portion_data[:-1], portion_data[1:]):
            sprite = pyglet.sprite.AdvancedSprite(hold_texture, batch=batch, program=skew_program)
            vertices = (
                last_portion[0], last_portion[2] + 93 - 1, 0, # bl
                last_portion[1], last_portion[2] + 93 - 1, 0, # br
                this_portion[1], this_portion[2] + 93 + 1, 0, # ur
                this_portion[0], this_portion[2] + 93 + 1, 0, # ul
            )
            sprite._vertex_list.position[:] = vertices # type: ignore
            sprites.append(sprite)
        for mid in mids:
            mid_begin, mid_real_y, mid_data = mid
            mid_data = (mid_begin, *mid_data)
            mid_y = (mid_real_y - hold_data[0][1]) * y_multiplier + y
            if mid_y < judge_line_imaginary_y:
                if mid_data[:-1] not in activated:
                    activate_note(mid_data)
                continue
            if mid_y > 600 / 0.05 / 2.5:
                continue
            mid_ratio = (mid_y - last_y) / (this_y - last_y)
            if "hold_ease_in" in last_data[2][-1]:
                mid_ratio = mid_ratio ** 2
            elif "hold_ease_out" in last_data[2][-1]:
                mid_ratio = 1 - ((1 - mid_ratio) ** 2)
            mid_x_l = int(((this_x_l - last_x_l) * mid_ratio) + last_x_l)
            mid_x_r = int(((this_x_r - last_x_r) * mid_ratio) + last_x_r)
            sprite = pyglet.sprite.AdvancedSprite(mid_texture, x=(mid_x_l + mid_x_r) / 2 - (mid_texture.width) / 8, y=mid_y + 93 - (mid_texture.height / 8), batch=batch, program=skew_program)
            sprite.scale = 0.25
            sprites.append(sprite)
        mids = []
        last_data = data


@window.event
def on_draw():
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    window.clear()
    
    sx, sy = aspect(textures["bg_default"], (1280, 720))
    textures["bg_default"].blit(640 - sx / 2, 360 - sy / 2, width=sx, height=sy)
    sx, sy = aspect(textures["lane_base"], (1280, 600))
    textures["lane_base"].blit(640 - sx / 2, 300 - sy / 2, width=sx, height=sy)
    sx, sy = aspect(textures["lane_line"], (1280, 600))
    textures["lane_line"].blit(640 - sx / 2, 300 - sy / 2, width=sx, height=sy)
    sx, sy = aspect(textures["judge_line"], (845, 60))
    textures["judge_line"].blit(640 - sx / 2, 120, width=sx, height=sy)

    global offset_time
    global previous_time
    current_time = time.perf_counter()

    if pause:
        if keys[key.DOWN]:
            offset_time = max(-1, offset_time - 0.05)
        elif keys[key.UP]:
            offset_time += 0.05
    else:
        offset_time += current_time - previous_time

    global music_player
    if not music_player and not pause and music_offset + offset_time > 0:
        music.seek(music_offset + offset_time - judge_line_offset_s)
        music_player = music.play()

    if offset_time < 0:
        offset_y = offset_time
        speed = 1
        offset_end_y = offset_y + (600 / 0.05 / 2.5) / y_multiplier
    else:
        base_time, offset_y, speed = y_lookup[bisect.bisect_right(y_lookup, offset_time, key=lambda e: e[0]) - 1]
        offset_y += (offset_time - base_time) * speed
        offset_end_y = offset_y + (600 / 0.05 / 2.5) / y_multiplier
    if offset_end_y < 0:
        offset_end_time = offset_end_y
    else:
        base_offset_end_y, offset_end_time, end_speed = inverse_y_lookup[bisect.bisect_right(inverse_y_lookup, offset_end_y, key=lambda e: e[0]) - 1]
        offset_end_time += (offset_end_y - base_offset_end_y) / end_speed

    global sprites
    global batch
    sprites = []
    batch = pyglet.graphics.Batch()
    for e in sorted(t[offset_time:offset_end_time]):
        begin = e.begin
        end = e.end
        real_y, data = e.data
        y = (real_y - offset_y) * y_multiplier
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
                        # hold path
                        draw_hold(hold_data, y)
                        # start / end
                        for begin, real_y, data in [hold_data[0], hold_data[-1]]:
                            head_y = (real_y - offset_y) * y_multiplier
                            if head_y < 0 or head_y > 600 / 0.05 / 2.5:
                                continue
                            draw_note((begin, *data), head_y)
                    case _:
                        # tap
                        draw_note((begin, *data[1]), y)

    batch.draw()

    activated.add(True)
    previous_time = current_time

@window.push_handlers
def on_key_press(symbol, modifiers):
    global pause
    global music_player
    if symbol == key.SPACE:
        pause = not pause
        if not pause:
            if music_offset + offset_time > 0:
                if music_player:
                    music_player.delete()
                music.seek(music_offset + offset_time - judge_line_offset_s)
                music_player = music.play()
        else:
            activated = set()
            if music_player:
                music_player.delete()
                music_player = None

pyglet.app.run()
