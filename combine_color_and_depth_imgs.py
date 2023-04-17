import os
from PIL import Image

"""
INPUT
```
{input_dir}
├── ngp_0050_0001_depth.png
├── ngp_0050_0001.png
├── ngp_0050_0002_depth.png
├── ngp_0050_0002.png
├── ngp_0050_0003_depth.png
├── ngp_0050_0003.png
├── ngp_0050_0004_depth.png
├── ngp_0050_0004.png
├── ngp_0050_0005_depth.png
├── ngp_0050_0005.png
├── ngp_0050_0006_depth.png
├── ngp_0050_0006.png
├── ngp_0050_0007_depth.png
├── ngp_0050_0007.png
├── ngp_0050_0008_depth.png
├── ngp_0050_0008.png
├── ngp_0050_0009_depth.png
├── ngp_0050_0009.png
...
```

OUTPUT
```
output_dir
├── ngp_0050_0001_combined.png
├── ngp_0050_0002_combined.png
├── ngp_0050_0003_combined.png
├── ngp_0050_0004_combined.png
├── ngp_0050_0005_combined.png
├── ngp_0050_0006_combined.png
├── ngp_0050_0007_combined.png
```

"""

input_dir = "/juno/u/tylerlum/github_repos/nerf_grasping/torch-ngp/isaac_Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682458/validation/"
output_dir = "/juno/u/tylerlum/github_repos/nerf_grasping/torch-ngp/isaac_Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682458/validation_combined/"

if os.path.exists(output_dir):
    print(f"WARNING: Output directory {output_dir} already exists. Exiting...")
    exit()

os.makedirs(output_dir)

depth_img_filenames = [filename for filename in os.listdir(input_dir) if filename.endswith("_depth.png")]
for depth_img_filename in depth_img_filenames:
    # Check if can find associated color image
    color_img_filename = depth_img_filename.replace("_depth.png", ".png")

    if not os.path.exists(os.path.join(input_dir, color_img_filename)):
        print(f"WARNING: Could not find color image for depth image {depth_img_filename}")
        continue

    # Combine the png images horizontally into a new png
    images = [Image.open(os.path.join(input_dir, depth_img_filename)), Image.open(os.path.join(input_dir, color_img_filename))]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_filename = depth_img_filename.replace("_depth.png", "_combined.png")
    print(f"Saving {new_filename}")
    new_im.save(os.path.join(output_dir, new_filename))

