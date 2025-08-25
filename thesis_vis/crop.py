import os
from PIL import Image

def crop_images_in_folder(main_folder):
    """
    Crops gt.png and imgs/pred_end.png in all subfolders of `main_folder` 
    by removing 20% from each side and overwrites them.
    """
    for root, dirs, files in os.walk(main_folder):
        for file_name in files:
            if file_name not in ["same_init.png", ".DS_Store"]:
                file_path = os.path.join(root, file_name)
                
                # Open image
                img = Image.open(file_path)
                w, h = img.size
                
                # Ensure it's square (optional check)
                if w != h:
                    print(f"Warning: {file_path} is not square ({w}x{h})")
                
                # Calculate cropping margins
                crop_margin = int(0.175 * w)  # 20% of width
                left = crop_margin
                top = crop_margin
                right = w - crop_margin
                bottom = h - crop_margin
                
                # Crop and overwrite
                cropped = img.crop((left, top, right, bottom))
                cropped.save(file_path)
                print(f"Cropped and saved: {file_path}")

# Example usage:
crop_images_in_folder("/Users/cedimac/Desktop/real_bones_thesis/color_bonesReal_PF++_noNoise_turn")
