from PIL import Image
import os
import random

def is_image_file(filename):
    """Determine whether a file is an image file"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # List of supported image file extensions
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def random_crop(img, size=(256, 256)):
    """Randomly crop a region of the specified size from the given image"""
    width, height = img.size
    crop_width, crop_height = size

    if width < crop_width or height < crop_height:
        return None  # Return None if the image size is smaller than the crop size

    x_left = random.randint(0, width - crop_width)
    y_upper = random.randint(0, height - crop_height)

    return img.crop((x_left, y_upper, x_left + crop_width, y_upper + crop_height))

# Folder paths (modify according to actual situation)
single_object_folder = './data/FSC147/box'
multiple_objects_folder = './data/FSC147/images_384_VarV2'
output_folder = './data/FSC147/one'

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_txt_path = os.path.join(output_folder, 'labels.txt')
with open(output_txt_path, 'w') as f:
    for folder, label in [(single_object_folder, 'one'), (multiple_objects_folder, 'more')]:
        for filename in os.listdir(folder):
            if is_image_file(filename):  # Only process image files
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path)

                # Save the original image and record it to the txt file
                original_img_output_path = os.path.join(output_folder, filename)
                img.save(original_img_output_path)
                f.write(f"{filename},{label}\n")

                # Randomly crop from the original image and save the cropped images
                for size in [(256, 384), (256, 256), (384, 384),(128,256),(256,128)]:
                    img_cropped = random_crop(img, size=size)
                    if img_cropped:
                        cropped_img_output_path = os.path.join(output_folder, f"{filename[:-4]}_random_{size[0]}x{size[1]}.jpg")
                        img_cropped.save(cropped_img_output_path)
                        f.write(f"{filename[:-4]}_random_{size[0]}x{size[1]}.jpg,{label}\n")

print("Dataset preparation complete.")
