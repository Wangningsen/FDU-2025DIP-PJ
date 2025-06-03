import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import os.path as osp

def detect_flare_source_on_image(img_resized):
    """
    Detects potential flare light sources on a resized BGR image by
    brightness thresholding and morphological operations,
    without requiring specific streak patterns.

    Args:
        img_resized (numpy.ndarray): The BGR image, typically resized to 512x512.

    Returns:
        list: A list of (cx, cy) coordinates for detected light sources
              in the 512x512 dimension.
    """
    if img_resized is None:
        return []

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 亮度阈值化：提取高亮区域
    # 阈值 200 用于捕获较亮的区域，可根据图片亮度进一步调整
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作：去除小噪声点，连接接近的亮斑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 先闭运算 (CLOSE) 连接断裂部分，再开运算 (OPEN) 去除孤立的小点
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # 查找所有可能光源的轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flare_sources_resized = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤掉面积过小（可能是残余噪声）或过大（可能是大面积非光源区域）的区域
        # 这些面积范围可能需要根据你的图片进行调整，现在设置为10到1000
        if area < 10 or area > 1000:
            continue

        # 计算中心点
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 直接添加中心点，不再进行条带检测
        flare_sources_resized.append((cx, cy))
            
    return flare_sources_resized


def process_folder_for_flares(input_folder, output_json_path, resized_output_folder=None, marked_output_folder=None):
    """
    Iterates through images in a folder, detects flare light sources,
    and saves results to a JSON file. It can optionally save resized images
    and images with detected light sources marked.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_json_path (str): Path for the output JSON file.
        resized_output_folder (str, optional): If provided, resized 512x512 images
                                                 will be saved here with the same filenames (as PNG).
        marked_output_folder (str, optional): If provided, images with detected light
                                                 sources marked will be saved here (as PNG).
    """
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG'} # Added .webp, .JPG, .JPEG
    
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Create output folders if they are specified and don't exist
    if resized_output_folder:
        os.makedirs(resized_output_folder, exist_ok=True)
        print(f"Resized images will be saved to: {resized_output_folder}")
    if marked_output_folder:
        os.makedirs(marked_output_folder, exist_ok=True)
        print(f"Marked images will be saved to: {marked_output_folder}")

    all_results = {}
    
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in supported_extensions]

    if not image_files:
        print(f"Error: No supported image files found in '{input_folder}'.")
        return
        
    print(f"Starting to process {len(image_files)} images in '{input_folder}'...")

    for filename in tqdm(image_files, desc="Processing progress"):
        image_path = os.path.join(input_folder, filename)
        
        img_original = cv2.imread(image_path)
        if img_original is None:
            print(f"\nWarning: Could not read image {filename}, skipping.")
            continue

        h_orig, w_orig, _ = img_original.shape
        
        # --- New Resizing and Cropping Logic ---
        target_size = 512
        
        # Determine the shorter side and calculate crop coordinates
        if h_orig < w_orig:
            crop_size = h_orig
            start_x = (w_orig - h_orig) // 2
            start_y = 0
        else:
            crop_size = w_orig
            start_x = 0
            start_y = (h_orig - w_orig) // 2
        
        # Crop the image to a square
        img_cropped = img_original[start_y:start_y + crop_size, start_x:start_x + crop_size]

        # Resize the cropped image to target_size x target_size
        interpolation_method = cv2.INTER_LINEAR if crop_size < target_size else cv2.INTER_AREA
        img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=interpolation_method)

        # Save the resized image if the folder is specified
        if resized_output_folder:
            base_filename_without_ext = os.path.splitext(filename)[0]
            resized_save_path = os.path.join(resized_output_folder, f"{base_filename_without_ext}.png")
            cv2.imwrite(resized_save_path, img_resized)

        resized_coords = detect_flare_source_on_image(img_resized)
        
        # --- Visualization Logic ---
        if marked_output_folder:
            # Create a copy for drawing marks
            img_display = img_resized.copy() 
            for cx, cy in resized_coords:
                # 绘制荧光绿色实心圆圈
                cv2.circle(img_display, (cx, cy), 15, (0, 255, 0), 2) # BGR: (Blue, Green, Red)

            # Save the image with marks as PNG
            base_filename_without_ext = os.path.splitext(filename)[0]
            marked_save_path = os.path.join(marked_output_folder, f"marked_{base_filename_without_ext}.png")
            cv2.imwrite(marked_save_path, img_display)


        # If no light sources are detected, store an empty list
        if not resized_coords:
            all_results[filename] = []
            continue

        # Convert coordinates from 512x512 back to original size
        # The scaling factor needs to consider the cropping and then resizing
        # The coordinates found are relative to the 512x512 resized *cropped* image.
        # We need to map them back to the original full image dimensions.
        
        # Scale back to cropped size first
        scale_to_cropped = crop_size / target_size
        
        original_coords = []
        for x_resized, y_resized in resized_coords:
            # Coordinates in the cropped section of the original image
            x_in_cropped_orig = int(x_resized * scale_to_cropped)
            y_in_cropped_orig = int(y_resized * scale_to_cropped)
            
            # Add back the offset from the original image for the crop
            x_original = x_in_cropped_orig + start_x
            y_original = y_in_cropped_orig + start_y
            
            original_coords.append((x_original, y_original))
            
        all_results[filename] = original_coords

    # Save all results to the JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"\nProcessing complete! All results saved to: {output_json_path}")
    except IOError as e:
        print(f"\nError: Could not write JSON file. {e}")


# --- Main execution block ---
if __name__ == '__main__':
    # --- Configure your paths here ---
    # 1. Input folder: Contains your original images
    input_directory = '/home/user2/wns/deflare/new_dataset' 

    # 2. Output JSON file: Stores the detected light source coordinates (in original image dimensions)
    output_json_file = 'new_dataset.json'

    # 3. Resized output folder (Optional): Where 512x512 resized images will be saved (as PNG).
    #    Set to None if you don't want to save resized images.
    resized_output_folder = './resized_new_dataset' 

    # 4. Marked output folder (Optional): Where images with marked light sources will be saved (as PNG).
    #    Set to None if you don't want to save marked images.
    marked_output_folder = './marked_light_resized_new_dataset' 

    # --- Run the main function ---
    process_folder_for_flares(
        input_directory,
        output_json_file,
        resized_output_folder=resized_output_folder,
        marked_output_folder=marked_output_folder
    )