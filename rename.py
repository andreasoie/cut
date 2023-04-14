import os

image_folder = "cherries"
output_folder = os.path.join(image_folder, "renamed")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

start_value = 3000
step = 3000

for idx, image in enumerate(range(start_value, 1131001, step), start=1):
    old_image_path = os.path.join(image_folder, f"{image}.png")
    new_image_path = os.path.join(output_folder, f"img{idx:04d}.png")

    if os.path.exists(old_image_path):
        os.rename(old_image_path, new_image_path)
