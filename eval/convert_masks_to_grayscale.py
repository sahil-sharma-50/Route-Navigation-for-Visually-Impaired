import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


# convert RGB labels to grayscale
def convert_rgb_to_gray(json_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Load color-label mapping from JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    rgb2gray = {}
    # Create a mapping from RGB color to label index
    for i, label in enumerate(data["labels"]):
        rgb = tuple(label["color"])
        rgb2gray[rgb] = i

    # Iterate over image files in the 'labels' directory
    for filename in tqdm(os.listdir(os.path.join(input_dir, "labels"))):
        if filename.endswith(".png"):
            # Open the RGB image and create a new grayscale image
            img = Image.open(os.path.join(input_dir, "labels", filename))
            img_rgb = img.convert("RGB")
            img_gray = Image.new("L", img.size)
            pixels_rgb = img_rgb.load()
            pixels_gray = img_gray.load()
            # Convert each pixel from RGB to grayscale based on the mapping
            for y in range(img.height):
                for x in range(img.width):
                    rgb = pixels_rgb[x, y]
                    gray = rgb2gray.get(rgb, 255)
                    pixels_gray[x, y] = gray
            # Adjust pixel values and save the grayscale image
            img_gray_array = np.array(img_gray)
            img_gray_array += 1
            img_gray = Image.fromarray(img_gray_array)
            img_gray.save(os.path.join(output_dir, filename))



if __name__ == "__main__":
    json_path = "Mapillary-Vistas-1000-sidewalks/config.json"
    base_dir = "Mapillary-Vistas-1000-sidewalks"
    output_base_dir = "Mapillary_converted_masks"
    for split in ["training", "validation"]:
        input_dir = os.path.join(base_dir, split)
        output_dir = os.path.join(output_base_dir, split)
        print(f"Converting {split} labels to gray scale in {output_dir}.....")
        convert_rgb_to_gray(json_path, input_dir, output_dir)
