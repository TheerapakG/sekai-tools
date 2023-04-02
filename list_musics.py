from pjsekai.models.master_data import MusicCategory, MusicDifficulty, MusicDifficultyType, ReleaseCondition, ResourceBox, ResourceBoxPurpose, ResourceBoxType, ResourceType
from pjsekai.enums.enums import MusicDifficultyType
from pjsekai.enums.unknown import Unknown
from pjsekai.client import Client
import datetime
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

CATEGORY = {MusicCategory.MV_2D.value: "2d", MusicCategory.MV.value: "3d", MusicCategory.ORIGINAL.value: "original"}

print("starting client ...")
client = Client(
    bytes(os.environ["KEY"], encoding="utf-8"),
    bytes(os.environ["IV"], encoding="utf-8"),
    system_info_file_path="./system-info.json",
    master_data_file_path="./master-data.json",
    user_data_file_path="./user-data.json",
    asset_directory="./asset",
)
print("updating data ...")
client.update_all()

difficulties_dict: dict[int, dict[MusicDifficultyType | Unknown, MusicDifficulty]] = {}
if difficulties := client.master_data.music_difficulties:
    for difficulty in difficulties:
        if (music_id := difficulty.music_id) and (music_difficulty := difficulty.music_difficulty):
            difficulty_types_dict = difficulties_dict.get(music_id, {})
            difficulty_types_dict[music_difficulty] = difficulty
            difficulties_dict[music_id] = difficulty_types_dict

release_conditions_dict: dict[int, ReleaseCondition] = {}
if release_conditions := client.master_data.release_conditions:
    for condition in release_conditions:
        if (condition_id := condition.id):
            release_conditions_dict[condition_id] = condition

music_resource_boxes_dict: dict[int, ResourceBox] = {}
if resource_boxes := client.master_data.resource_boxes:
    for resource_box in resource_boxes:
        if resource_box.resource_box_purpose == ResourceBoxPurpose.SHOP_ITEM and resource_box.resource_box_type == ResourceBoxType.EXPAND:
            if details := resource_box.details:
                for detail in details:
                    if (detail.resource_type == ResourceType.MUSIC) and (resource_id := detail.resource_id):
                        music_resource_boxes_dict[resource_id] = resource_box

TAG = {
    "vocaloid": "vir",
    "light_music_club": "leo",
    "idol": "mor",
    "street": "viv",
    "theme_park": "won",
    "school_refusal": "25j",
    "other": "oth",
}

music_tags: defaultdict[int, set[str]] = defaultdict(set)
if tags := client.master_data.music_tags:
    for tag in tags:
        if (music_id := tag.music_id) and (music_tag := tag.music_tag):
            music_tags[music_id].add(music_tag)

if musics := client.master_data.musics:
    unreleased_musics = filter(lambda music: music.published_at and music.published_at > datetime.datetime.now(datetime.timezone.utc), musics)
    # unreleased_musics = musics
    for unreleased_music in sorted(unreleased_musics, key=lambda music: music.published_at if music.published_at else -1):
        music_published_at = f"{unreleased_music.published_at.astimezone().date()} {unreleased_music.published_at.astimezone().time()}" if unreleased_music.published_at else None
        music_id = unreleased_music.id
        music_categories = [category for category in unreleased_music.categories if category != MusicCategory.IMAGE] if unreleased_music.categories else []
        music_categories_str = f"[{'|'.join([CATEGORY[category.value] for category in music_categories if category.value and CATEGORY.get(category.value)])}]"if music_categories else ""
        music_tags_str = f"[{'|'.join([name for tag, name in TAG.items() if tag in music_tags[music_id]])}]" if music_id and music_tags[music_id] else None
        music_resource_box = music_resource_boxes_dict.get(music_id) if music_id else None
        music_resource_box_id = music_resource_box.id if music_resource_box else None
        music_ids_str = " ".join([s for s in [f'm: {music_id}' if music_id else '', f'r: {music_resource_box_id}' if music_resource_box_id else ''] if s])
        music_difficulties = difficulties_dict.get(music_id, {}) if music_id else {}
        music_difficulty_inner_str = "/".join(map( \
            lambda music_difficulty: str(music_difficulty.play_level) if music_difficulty and music_difficulty.play_level else '??', \
            map(lambda difficulty: music_difficulties.get(difficulty), MusicDifficultyType) \
        ))
        music_difficulty_str = f"LV: {music_difficulty_inner_str}"
        music_condition_id = unreleased_music.release_condition_id
        music_condition_str = None
        match music_condition_id:
            case None:
                music_condition_str = "??"
            case 1:
                music_condition_str = "unlocked"
            case 5:
                music_condition_str = "shop: 2"
            case 10:
                music_condition_str = "present"
            case _:
                music_condition = release_conditions_dict.get(music_condition_id)
                music_condition_str = f"{music_condition_id}: {music_condition.sentence}" if music_condition else "??"
        out_str = " | ".join([
            " ".join([
                s 
                for s 
                in [
                    f"**{unreleased_music.title}**",
                    music_categories_str,
                    music_tags_str,
                    str(music_published_at)
                ]
                if s
            ]),
            music_difficulty_str,
            music_ids_str,
            music_condition_str
        ])
        print(out_str)
