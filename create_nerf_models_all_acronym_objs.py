import subprocess
import os
from tqdm import tqdm

"""
Create nerf models for all objects in the acronym dataset

For example:
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/isaac_NintendoDS_1ce1db5e6f9b9accf3ddd47627ca960_0.0378025115 --workspace ../nerf_checkpoints/isaac_NintendoDS_1ce1db5e6f9b9accf3ddd47627ca960_0.0378025115/ --cuda_ray --bound 2 --scale 1.0 --mode blender --fp16 --max_epochs 100
```
"""


acronym_object_classes = sorted([folder for folder in os.listdir("data") if folder.startswith("isaac") and '.' in folder])
print(f"Found {len(acronym_object_classes)} objects in the acronym dataset")
print(f"First 10: {acronym_object_classes[:10]}")

num_failed = 0

indices = list(range(len(acronym_object_classes)))
RANDOMIZE_ORDER = True
if RANDOMIZE_ORDER:
    print(f"Randomizing order...")
    import random
    random.shuffle(indices)

for i in (pbar := tqdm(indices)):
    acronym_object_class = acronym_object_classes[i]
    pbar.set_description(f"num_failed = {num_failed}")

    output_path = f"../nerf_checkpoints/{acronym_object_class}"
    if os.path.exists(output_path):
        print(f"output_path = {output_path} already exists, skipping")
        continue

    try:
        command = f"OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/{acronym_object_class} --workspace {output_path} --cuda_ray --bound 2 --scale 1.0 --mode blender --fp16 --max_epochs 100"
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"e = {e}")
        num_failed += 1
        print("Continuing")

