import subprocess
import os
from tqdm import tqdm

"""
Create nerf models for all objects in the acronym dataset

For example:
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/isaac_NintendoDS_1ce1db5e6f9b9accf3ddd47627ca960_0.0378025115 --workspace ../nerf_checkpoints/isaac_NintendoDS_1ce1db5e6f9b9accf3ddd47627ca960_0.0378025115/ --cuda_ray --bound 2 --scale 1.0 --mode blender --fp16
```
"""


acronym_object_classes = [folder for folder in os.listdir("data") if folder.startswith("isaac") and '.' in folder]
print(f"Found {len(acronym_object_classes)} objects in the acronym dataset")
print(f"First 10: {acronym_object_classes[:10]}")

num_failed = 0
for acronym_object_class in (pbar := tqdm(acronym_object_classes)):
    pbar.set_description(f"num_failed = {num_failed}")

    try:
        command = f"OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/{acronym_object_class} --workspace  ../nerf_checkpoints/{acronym_object_class} --cuda_ray --bound 2 --scale 1.0 --mode blender --fp16"
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"e = {e}")
        num_failed += 1
        print("Continuing")

