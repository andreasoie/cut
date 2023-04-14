import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

matplotlib.use("TkAgg")  # You can replace "TkAgg"

import random


class RandomRotateAndZoom:
    def __init__(self, degrees, fill=None, fillcolor=None):
        self.degrees = degrees
        self.fill = fill
        self.fillcolor = fillcolor

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        rotated_img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=self.fillcolor)

        # Calculate the zoom factor needed to prevent black bars
        angle_rad = abs(angle) * (3.14159 / 180)
        zoom_factor = 1 / (1 - (1 - abs(angle) / 90) * 0.5)

        width, height = img.size
        new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
        rotated_img = rotated_img.resize((new_width, new_height), resample=Image.BICUBIC)

        paste_x, paste_y = (rotated_img.width - rotated_img.width) // 2, (rotated_img.height - rotated_img.height) // 2
        rotated_img.paste(rotated_img, (paste_x, paste_y))

        return rotated_img


def main(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # resize = 256
    image = image.resize((256, 256))
    
    rotation_degrees = 10
    zoom_range = (0.9, 1.1)
    padding = 10

    # Define the transformation pipeline
    tfms = transforms.Compose([
        transforms.Pad(padding, fill=0, padding_mode='reflect'),
        RandomRotateAndZoom(degrees=10, fillcolor=(0, 0, 0)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Apply the transformationstransforms
    transformed_image = tfms(image)

    # Convert the transformed image back to PIL format
    transformed_image_pil = transforms.ToPILImage()(transformed_image)

    # Plot the original and transformed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(transformed_image_pil)
    axes[1].set_title("Transformed Image")
    axes[1].axis("off")
    plt.show()

if __name__ == "__main__":
    while True:
        for i in range(1, 6):
            imgdir = "/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry/study_cases_cherry/optical"
            image_path = imgdir + f"/object_{i}.png"
            main(image_path)