from pjsekai.client import Client
import UnityPy
import UnityPy.config
from UnityPy.tools.extractor import EXPORT_TYPES, export_obj

import shutil
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

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
print("logging in ...")
client.login(int(os.environ["USER_ID"]), os.environ["CREDENTIAL"])

UnityPy.config.FALLBACK_UNITY_VERSION = client.platform.unity_version

export_types_keys = list(EXPORT_TYPES.keys())

def defaulted_export_index(type):
    try:
        return export_types_keys.index(type)
    except (IndexError, ValueError):
        return 999
    
directory = client.asset_directory
if not directory:
    raise ValueError("no asset directory")

asset = client.asset
if not asset:
    raise ValueError("no asset")

if not (asset.asset_bundle_info and asset.asset_bundle_info.version == asset.version):
    print("updating asset bundle info ...")
    asset_bundle_info = asset.get_asset_bundle_info(client.api_manager)
else:
    print("asset bundle info already updated ...")
    asset_bundle_info = asset.asset_bundle_info

bundles = asset_bundle_info.bundles
if not bundles:
    raise ValueError("no bundles")


def load_asset(asset_bundle_str: str, force: bool = False) -> bool:
    bundle_hash = bundles[asset_bundle_str].hash
    if not bundle_hash:
        print(f"no bundle for {asset_bundle_str}")
        return False
    
    if (directory / "bundle" / f"{asset_bundle_str}.unity3d").exists():
        try:
            with open(directory / "hash" / asset_bundle_str, "r") as f:
                if f.read() == bundle_hash and not force:
                    print(f"bundle {asset_bundle_str} already updated")
                    return False
        except FileNotFoundError:
            pass
        
        print(f"updating bundle {asset_bundle_str}")
    else:
        print(f"downloading bundle {asset_bundle_str}")

    with client.api_manager.download_asset_bundle(asset_bundle_str) as asset_bundle:
        (directory / "bundle" / asset_bundle_str).parent.mkdir(parents=True, exist_ok=True)
        with open(directory / "bundle" / f"{asset_bundle_str}.unity3d", "wb") as f:
            for chunk in asset_bundle.chunks:
                f.write(chunk)

        env = UnityPy.load(str(directory / "bundle" / f"{asset_bundle_str}.unity3d"))
        container = sorted(
            env.container.items(), key=lambda x: defaulted_export_index(x[1].type)
        )

        for obj_path, obj in container:
            obj_path = "/".join(x for x in obj_path.split("/") if x)
            obj_dest = directory / obj_path
            obj_dest.parent.mkdir(parents=True, exist_ok=True)

            print(f"extracting {obj_path}")

            export_obj(
                obj, # type: ignore
                obj_dest,
            )

            if obj_dest.suffixes == [".acb", ".bytes"]:
                shutil.copy(obj_dest, obj_dest.with_suffix(""))
                subprocess.run(["./vgmstream-cli", "-o", obj_dest.parent / "?n.wav", "-S", "0", obj_dest.with_suffix("")])
    
    (directory / "hash" / asset_bundle_str).parent.mkdir(parents=True, exist_ok=True)
    with open(directory / "hash" / asset_bundle_str, "w") as f:
        f.write(bundle_hash)
            
    print(f"updated bundle {asset_bundle_str}")
    return True

if __name__ == "__main__":
    master_data = client.master_data
    if not master_data:
        raise ValueError("no master data")
    
    _musics = master_data.musics
    if not _musics:
        raise ValueError("no musics")
    musics = {music.id: music for music in _musics if music.id}
    
    music_vocals = master_data.music_vocals
    if not music_vocals:
        raise ValueError("no music vocals")
    
    asset_bundles =  []

    asset_bundles.extend([
        "shader/common",
        "shader/live",
        "shader/particles",
        "shader/rp_common",
        "shader/sandbox",
        "effect_asset/live/skill/default",
        "effect_asset/live/tap_effect/default",
        "live/2dmode/background/default",
        "live/note/default",
        "live/tap_se/default",
        "music/long/se_0333_01",
        "music/long/se_0321_01"
    ])
    
    asset_bundles.extend((bundle.bundle_name 
        for bundle in bundles.values() 
        if bundle.bundle_name and \
            bundle.bundle_name.startswith("music/music_score")
    ))

    for vocal in music_vocals:
        if vocal.seq == 1:
            if asset_bundle_name := vocal.asset_bundle_name:
                asset_bundles.append(f"music/long/{asset_bundle_name}")
            else:
                print(f"cannot find vocal bundle for {musics.get(vocal.music_id, f'music id {vocal.music_id}') if vocal.music_id else 'unknown music'}")

    updated_bundles = [*filter(load_asset, asset_bundles)]

    print(f"updated {len(updated_bundles)} bundles")
    for bundle in updated_bundles:
        print(f" - {bundle}")
