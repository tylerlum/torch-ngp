import subprocess
import os
from tqdm import tqdm
from termcolor import colored
from typing import DefaultDict


# INPUT PARAMS
path_to_input_dir = "/juno/u/tylerlum/github_repos/nerf_grasping/torch-ngp/data/krishnan_data"
input_dirs = [
    "big_banana",
    "isaac_banana",
    "isaac_box",
    "isaac_dec_banana",
    "isaac_dec_bleach_cleanser",
    "isaac_dec_teddy_bear",
    "isaac_empty",
    "isaac_power_drill",
    "isaac_teddy",
    "isaac_teddy_bear",
    "isaac_teddy_bear2",
    "nerf_banana",
    "nerf_llff_data",
    "nerf_power_drill",
    "nerf_synthetic",
    "nerf_teddy_bear",
    "new_banana",
    "new_bleach_cleanser",
    "new_box",
    "new_power_drill",
    "new_teddy_bear",
    "teddy_bear_dataset",
]

print("=" * 100)
print("PARAMS")
print("=" * 100)
print(f"path_to_input_dir: {path_to_input_dir}")
print(f"input_dirs: {input_dirs}")
print()

for input_dir in tqdm(input_dirs):
    try:
        full_input_dir = os.path.join(path_to_input_dir, input_dir)
        command = f"OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py {full_input_dir} --workspace {input_dir + '_2023-04-14'} --cuda_ray --bound 2 --scale 1.0 --mode blender --fp16"

        print(f"command: {command}")
        subprocess.run(command, shell=True, check=True)
        print()
    except subprocess.CalledProcessError as e:
        print("=" * 100)
        print(
            colored(
                f"Error: {e} when processing {input_dir}. Skipping it."
            )
        )
        print("=" * 100)
        print()
