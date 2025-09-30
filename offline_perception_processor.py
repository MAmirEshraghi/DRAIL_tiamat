"""
independent script to run the perception pipeline on pre-recorded data
from an obs_buffer.pkl file.
"""
import os
import json
import numpy as np
import datetime
import shutil
import time
import pickle
import torch
import cv2
from PIL import Image
import csv
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from transformers import AutoProcessor, AutoModelForVision2Seq
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Import the data buffer classes 
from obs_data_buffer import ObsDataBuffer, ObsDataEntry, compose_transforms_optimized

# =====================================================================================
#  1. Data Structures & AI Helpers
# =====================================================================================


class PerceptionLog:
    """Manages the creation, organization, and saving of perception data."""
    def __init__(self, args, base_dir="logs/perception_logs_offline"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # get model name from the VLM ID
        vlm_name_abbr = args.VLM_MODEL_ID.split('/')[-1]

        # a clear string for the process limit (e.g., lim150 or limAll)
        limit_str = f"lim{args.PROCESS_LIMIT}" if args.PROCESS_LIMIT > 0 else "limAll"
        
        # build descriptive string 
        param_str = (
            f"ct{args.centroid_threshold}"
            f"_covt{str(args.coverage_threshold).replace('.', 'p')}"
            f"_vs{str(args.voxel_size).replace('.', 'p')}"
            f"_vlmp{str(args.vlm_padding).replace('.', 'p')}"
            f"_mma{args.min_mask_area}"
            f"_{limit_str}"
            f"_vlm{vlm_name_abbr}"
        )

        self.scan_id = f"{timestamp}_{param_str}"
        self.scan_dir = os.path.join(base_dir, self.scan_id)

        self.image_counter = 0
        self.instance_counter = 0

        os.makedirs(self.scan_dir, exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "obj_frame_visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "frame_visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "object_visualization"), exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "masks_png"), exist_ok=True)
        os.makedirs(os.path.join(self.scan_dir, "merge_tracking"), exist_ok=True)

        print(f"[PerceptionLog] Initialized new scan log at: {self.scan_dir}")

        self.data = {
            "scan_metadata": {
                "scan_id": self.scan_id,
                "start_time": datetime.datetime.now().isoformat(),
                "source_file": None,
            },
            "images": {},
            "unique_objects": {},
            "object_instances": {}
        }

    def add_image(self, rgb_np_0_255, depth_np, camera_pose_7d):
        """Adds a new image capture from NumPy data and saves it."""
        image_id = f"img_{self.image_counter:03d}"
        
        # Save RGB Image
        rgb_path = os.path.join(self.scan_dir, "rgb", f"{image_id}.png")
        Image.fromarray(rgb_np_0_255).save(rgb_path)

        # Save Depth Data
        depth_path = os.path.join(self.scan_dir, "depth", f"{image_id}.npy")
        np.save(depth_path, depth_np)
        
        self.data["images"][image_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "camera_pose_world": camera_pose_7d.tolist(),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "detected_object_instances": []
        }
        self.image_counter += 1
        return image_id

    def add_object_instance(self, image_id, object_id, bbox, mask_area, mask_np):
        instance_id = f"inst_{self.instance_counter:04d}"
        mask_path = os.path.join(self.scan_dir, "masks", f"{instance_id}.npy")
        np.save(mask_path, mask_np)
        mask_image_np = (mask_np.astype(np.uint8) * 255)
        mask_png_path = os.path.join(self.scan_dir, "masks_png", f"{instance_id}.png")
        Image.fromarray(mask_image_np).save(mask_png_path)
        self.data["object_instances"][instance_id] = {
            "parent_image_id": image_id,
            "parent_object_id": object_id,
            "bounding_box": [int(c) for c in bbox],
            "mask_area": int(mask_area),
            "mask_path": mask_path
        }
        if object_id in self.data["unique_objects"]:
            self.data["unique_objects"][object_id]["instances"].append(instance_id)
            if image_id not in self.data["unique_objects"][object_id]["seen_in_images"]:
                self.data["unique_objects"][object_id]["seen_in_images"].append(image_id)
                   
        self.instance_counter += 1
        return instance_id

    def add_or_update_unique_object(self, object_id, description, world_position):
        if object_id not in self.data["unique_objects"]:
            self.data["unique_objects"][object_id] = {
                "vlm_description": description,
                "avg_world_position": world_position.tolist(),
                "instances": [],
                "seen_in_images": [] 
            }
        else:
            self.data["unique_objects"][object_id]["vlm_description"] = description
            self.data["unique_objects"][object_id]["avg_world_position"] = world_position.tolist()

    def finalize_and_save(self, source_file=None):
        self.data["scan_metadata"]["source_file"] = source_file
        log_path = os.path.join(self.scan_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"[PerceptionLog] Successfully saved log to: {log_path}")
    
    def add_visualization_path_to_object(self, object_id, viz_path):
        """Adds the file path of the best instance visualization to a unique object's data."""
        if object_id in self.data["unique_objects"]:
            self.data["unique_objects"][object_id]["best_instance_image_path"] = viz_path
    

# --- Helper Functions ---
def get_vlm_description(cropped_object_image, model, processor):
    try:
        pil_image = Image.fromarray(cropped_object_image).convert('RGB')  
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "This object musked as a part of things inside of a home. Describe the object based on the observation and your creativety shortly, what is that?. If the image is not describable, you must respond with 'unknown object'."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[pil_image], return_tensors="pt").to(model.device)
        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        description = processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
        return description.strip().lower()
    except Exception as e:
        print(f"[SmolVLM Error] {e}")
        return "unknown object"

def draw_detection_on_image(image_to_draw_on, mask_np, bbox, label_text, color_rgb):
    overlay_color = np.array(color_rgb, dtype=np.uint8)
    masked_area = image_to_draw_on[mask_np]
    blended_pixels = (masked_area * 0.5 + overlay_color * 0.5).astype(np.uint8)
    image_to_draw_on[mask_np] = blended_pixels
    x, y, w, h = bbox
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image_to_draw_on, (x, y), (x + w, y + h), color_bgr, 2)
    text_position = (x, y - 10 if y > 10 else y + 10)
    cv2.putText(img=image_to_draw_on, text=label_text, org=text_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)

def calculate_point_cloud_coverage(pcd1, pcd2, voxel_size=0.05):
    if len(pcd1) == 0 or len(pcd2) == 0: return 0.0
    pcd1_hashed = np.floor(pcd1 / voxel_size).astype(int)
    pcd2_hashed = np.floor(pcd2 / voxel_size).astype(int)
    pcd1_voxels = set(map(tuple, pcd1_hashed))
    pcd2_voxels = set(map(tuple, pcd2_hashed))
    if len(pcd1_voxels) == 0: return 0.0
    intersection_size = len(pcd1_voxels.intersection(pcd2_voxels))
    return intersection_size / len(pcd1_voxels)

def apply_non_max_suppression(raw_masks, iou_threshold=0.7):
    if not raw_masks: return []
    boxes = np.array([[m['bbox'][0], m['bbox'][1], m['bbox'][0] + m['bbox'][2], m['bbox'][1] + m['bbox'][3]] for m in raw_masks])
    scores = np.array([m['predicted_iou'] for m in raw_masks])
    indices = np.argsort(scores)[::-1]
    keep_indices = []
    while len(indices) > 0:
        current_index = indices[0]
        keep_indices.append(current_index)
        top_box = boxes[current_index]
        other_indices = indices[1:]
        other_boxes = boxes[other_indices]
        x1 = np.maximum(top_box[0], other_boxes[:, 0]); y1 = np.maximum(top_box[1], other_boxes[:, 1])
        x2 = np.minimum(top_box[2], other_boxes[:, 2]); y2 = np.minimum(top_box[3], other_boxes[:, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        union = (top_box[2] - top_box[0]) * (top_box[3] - top_box[1]) + (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1]) - intersection
        iou = intersection / union
        indices_to_remove = np.where(iou > iou_threshold)[0]
        indices = np.delete(indices, np.concatenate(([0], indices_to_remove + 1)))
    return [raw_masks[i] for i in keep_indices]

def create_object_visualization(cropped_image_np, label_text):
    padding, text_area_height, font_scale, font_thickness = 20, 50, 0.6, 2
    obj_h, obj_w, _ = cropped_image_np.shape
    (text_w, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    canvas_w = max(obj_w + (2 * padding), text_w + (2 * padding))
    canvas_h = obj_h + (2 * padding) + text_area_height
    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)
    paste_x = (canvas_w - obj_w) // 2
    paste_y = padding + text_area_height
    canvas[paste_y : paste_y + obj_h, paste_x : paste_x + obj_w] = cropped_image_np
    text_pos = (padding, padding + 20) 
    cv2.putText(img=canvas, text=label_text, org=text_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness)
    return canvas

def create_visualization(rgb_image_np, mask_np, bbox, description_text):
    # Define the fixed heights for the final image
    FIXED_IMAGE_HEIGHT = 256
    FINAL_CANVAS_HEIGHT = 430

    # create the annotated image with the mask and bbox
    viz_image = rgb_image_np.copy()
    green_color = np.array([0, 255, 0], dtype=np.uint8)
    masked_area = viz_image[mask_np]
    blended_pixels = (masked_area * 0.5 + green_color * 0.5).astype(np.uint8)
    viz_image[mask_np] = blended_pixels
    x, y, w, h = [int(c) for c in bbox] # ensure coordinates are integers
    cv2.rectangle(viz_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # calculate the new width to maintain the original image's aspect ratio
    orig_h, orig_w, _ = viz_image.shape
    aspect_ratio = orig_w / orig_h
    new_w = int(FIXED_IMAGE_HEIGHT * aspect_ratio)
    # Resize the annotated image to the fixed height
    resized_image = cv2.resize(viz_image, (new_w, FIXED_IMAGE_HEIGHT))

    # create Final Canvas with Fixed Height
    final_canvas = np.full((FINAL_CANVAS_HEIGHT, new_w, 3), 255, dtype=np.uint8) # White background
    # place the resized annotated image at the top of the canvas
    final_canvas[0:FIXED_IMAGE_HEIGHT, 0:new_w] = resized_image

    # text on Bottom Padding Logic 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 0, 0) # Black 
    line_height = 25
    padding_left_right = 10

    # wrap text based on the new canvas width
    words = description_text.split(' ')
    wrapped_lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        if text_width > (new_w - 2 * padding_left_right):
            wrapped_lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    wrapped_lines.append(current_line)

    # draw the wrapped text onto the white padding area
    text_y_start = FIXED_IMAGE_HEIGHT + padding_left_right + 15 # start text below the image
    for i, line in enumerate(wrapped_lines):
        text_position = (padding_left_right, text_y_start + i * line_height)
        cv2.putText(final_canvas, line, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return final_canvas

# --- Point Cloud Functions ---
def get_o3d_cam_intrinsic(height, width):
    """
    Returns the Open3D camera intrinsic object.
    Dynamically creates intrinsics based on image size.
    """
    # assuming 90-degree FOV, fx=fy=width/2, cx=width/2, cy=height/2
    fx = width / 2.0
    fy = width / 2.0  # Often fx=fy
    cx = width / 2.0
    cy = height / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def get_organized_point_cloud(depth_np, camera_pose_7d, camera_intrinsics):
    """
    Projects a depth image into a 3D point cloud, preserving the HxW structure.
    Returns a numpy array of shape (H, W, 3) in world coordinates.
    """
    H, W = depth_np.shape
    fx, fy, cx, cy = camera_intrinsics.intrinsic_matrix[0, 0], camera_intrinsics.intrinsic_matrix[1, 1], camera_intrinsics.intrinsic_matrix[0, 2], camera_intrinsics.intrinsic_matrix[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z_cam = depth_np
    X_cam = (u - cx) * Z_cam / fx
    Y_cam = (v - cy) * Z_cam / fy
    points_camera_frame = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
    points_flat = points_camera_frame.reshape(-1, 3)
    quat_wxyz, position_xyz = camera_pose_7d[:4], camera_pose_7d[4:]
    rotation_matrix = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix() # w,x,y,z -> x,y,z,w
    c2w_transform = np.eye(4)
    c2w_transform[:3, :3] = rotation_matrix
    c2w_transform[:3, 3] = position_xyz
    points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1))))
    points_world_homogeneous = (c2w_transform @ points_homogeneous.T).T
    return points_world_homogeneous[:, :3].reshape(H, W, 3)

def analyze_and_save_timing_log2(perception_log, timing_log):
    """Saves raw timing data, generates performance plots, and prints a detailed summary."""
    print("\n--- 5. Analyzing Performance and Generating Reports ---")
    
    timing_dir = os.path.join(perception_log.scan_dir, "timing_analysis")
    os.makedirs(timing_dir, exist_ok=True)

    json_path = os.path.join(timing_dir, "timing_log.json")
    with open(json_path, "w") as f:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        json.dump(timing_log, f, indent=4, cls=NumpyEncoder)
    print(f"  > Raw timing data saved to: {json_path}")

    for key, value in timing_log.items():
        if isinstance(value, list) and len(value) > 1:
            try:
                # adapt plot based on metric type
                y_data, y_label, unit = [], "", ""
                if "time_ms" in value[0]:
                    y_data = [d['time_ms'] for d in value]
                    y_label, unit = "Time (ms)", "ms"
                elif "mask_count" in value[0]:
                    y_data = [d['mask_count'] for d in value]
                    y_label, unit = "Number of Masks", ""
                else:
                    continue # skip if it's not a recognized format

                safe_filename = key.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "") + "_performance.png"
                plot_path = os.path.join(timing_dir, safe_filename)
                x_data = range(len(y_data))

                plt.figure(figsize=(15, 7))
                plt.plot(x_data, y_data, marker='o', linestyle='-', markersize=4, label=f'{y_label} per Sample')
                
                mean_val = np.mean(y_data)
                plt.axhline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}{unit}')
                
                plt.title(f"Performance for '{key}'", fontsize=16)
                plt.xlabel("Sample Index", fontsize=12)
                plt.ylabel(y_label, fontsize=12)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                print(f"  > Performance plot saved to: {plot_path}")
            except Exception as e:
                print(f"  > Could not generate plot for '{key}': {e}")
    
    print("\n\n" + "="*80)
    print(" " * 28 + "PERFORMANCE SUMMARY")
    print("="*80)
    for key, value in timing_log.items():
        if isinstance(value, list):
            if len(value) > 0:
                # --- Adapt summary based on metric type ---
                data_list, unit = [], ""
                if "time_ms" in value[0]:
                    data_list = [d['time_ms'] for d in value]
                    unit = "ms"
                elif "mask_count" in value[0]:
                    data_list = [d['mask_count'] for d in value]
                    unit = "" # No unit for a simple count

                stats = { "samples": len(data_list), "avg": np.mean(data_list), "std": np.std(data_list), "min": np.min(data_list), "median": np.median(data_list), "max": np.max(data_list) }
                print(f"{key:<35}: "
                      f"avg={stats['avg']:.3f}{unit}, std={stats['std']:.3f}{unit}, "
                      f"min={stats['min']:.3f}{unit}, max={stats['max']:.3f}{unit} "
                      f"({stats['samples']} samples)")
            else:
                print(f"{key:<35}: No data collected.")
        else:
            print(f"{key:<35}: {value*1000:.3f} ms")
    print("="*80)

def analyze_and_save_timing_log(perception_log, timing_log, process_limit):
    """Saves raw timing data, generates plots, and prints/saves a detailed summary."""
    print("\n--- 5. Analyzing Performance and Generating Reports ---")
    
    timing_dir = os.path.join(perception_log.scan_dir, "timing_analysis")
    os.makedirs(timing_dir, exist_ok=True)

    # (The JSON saving and plot generation part remains the same)
    json_path = os.path.join(timing_dir, "timing_log.json")
    with open(json_path, "w") as f:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        json.dump(timing_log, f, indent=4, cls=NumpyEncoder)
    print(f"  > Raw timing data saved to: {json_path}")

    for key, value in timing_log.items():
        if isinstance(value, list) and len(value) > 1:
            try:
                y_data, y_label, unit = [], "", ""
                if "time_ms" in value[0]:
                    y_data = [d['time_ms'] for d in value]
                    y_label, unit = "Time (ms)", "ms"
                elif "mask_count" in value[0]:
                    y_data = [d['mask_count'] for d in value]
                    y_label, unit = "Number of Masks", ""
                else: continue
                safe_filename = key.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "") + "_performance.png"
                plot_path = os.path.join(timing_dir, safe_filename)
                x_data = range(len(y_data))
                plt.figure(figsize=(15, 7))
                plt.plot(x_data, y_data, marker='o', linestyle='-', markersize=4, label=f'{y_label} per Sample')
                mean_val = np.mean(y_data)
                plt.axhline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}{unit}')
                plt.title(f"Performance for '{key}'", fontsize=16); plt.xlabel("Sample Index", fontsize=12)
                plt.ylabel(y_label, fontsize=12); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend(); plt.tight_layout(); plt.savefig(plot_path); plt.close()
                print(f"  > Performance plot saved to: {plot_path}")
            except Exception as e:
                print(f"  > Could not generate plot for '{key}': {e}")
    
    # --- Logic to build the summary, print it, and save it to a file ---
    summary_lines = []
    separator = "="*80
    summary_lines.append(separator)
    summary_lines.append(" " * 28 + "PERFORMANCE SUMMARY")
    summary_lines.append(separator)

    # Add the process limit information
    total_processed_count = len(timing_log.get("3a. SAM per image", []))
    if process_limit:
        limit_str = f"Image Process Limit: {process_limit}"
    else:
        limit_str = f"Image Process Limit: None (Processed {total_processed_count} total images)"
    summary_lines.append(f"{'Run Configuration':<35}: {limit_str}")
    summary_lines.append("-" * 80)

    # Add timing stats
    for key, value in timing_log.items():
        if isinstance(value, list):
            if len(value) > 0:
                data_list, unit = [], ""
                if "time_ms" in value[0]:
                    data_list = [d['time_ms'] for d in value]; unit = "ms"
                elif "mask_count" in value[0]:
                    data_list = [d['mask_count'] for d in value]; unit = ""
                stats = { "samples": len(data_list), "avg": np.mean(data_list), "std": np.std(data_list), "min": np.min(data_list), "median": np.median(data_list), "max": np.max(data_list) }
                line = (f"{key:<35}: "
                        f"avg={stats['avg']:.3f}{unit}, std={stats['std']:.3f}{unit}, "
                        f"min={stats['min']:.3f}{unit}, max={stats['max']:.3f}{unit} "
                        f"({stats['samples']} samples)")
                summary_lines.append(line)
            else:
                summary_lines.append(f"{key:<35}: No data collected.")
        else:
            summary_lines.append(f"{key:<35}: {value*1000:.3f} ms")
    
    summary_lines.append(separator)
    
    # Join all lines into a single string
    summary_text = "\n".join(summary_lines)

    # Print to console
    print("\n\n" + summary_text)

    # Save to file
    summary_path = os.path.join(perception_log.scan_dir, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n> Performance summary saved to: {summary_path}")

# ====================================================================================
# 2. MAIN OFFLINE PROCESSING SCRIPT
# =====================================================================================

def main(args):
    """
    Main function to load data, run the perception pipeline, and save results.
    Accepts a namespace object with configuration parameters.
    """
    # # --- Configuration ---
    # PKL_FILE_PATH = "tiamat_fsm_task_scripts/obs_buffer.pkl"
    # SAM_CHECKPOINT_PATH =   "tiamat_fsm_task_scripts/models/sam_vit_l_0b3195.pth" # "tiamat_fsm_task_scripts/models/sam_vit_b_01ec64.pth"  sam_vit_l_0b3195.pth
    # VLM_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Performance Timing Setup ---
    timing_log = {
        "Total Time": 0.0,
        "1. Model Loading": 0.0,
        "2. Data Ingestion": 0.0,
        "3. Full AI Processing": 0.0,
        "3a. SAM per image": [],
        "3b. Dedup Centroid Check per mask": [],
        "3c. Dedup Coverage Check per mask": [],
        "3d. VLM Description per object": [],
        "3e. Masks per image": [],
        "4. Visualization": 0.0
    }
    
    overall_start_time = time.perf_counter()
    
    # --- 1. Load AI Models ---
    print("--- 1. Loading AI Models ---")
    model_load_start = time.perf_counter()
    
    sam = sam_model_registry["vit_l"](checkpoint=args.SAM_CHECKPOINT_PATH)
    sam.to(args.DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=args.min_mask_area)
    
    processor = AutoProcessor.from_pretrained(args.VLM_MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        args.VLM_MODEL_ID, torch_dtype=torch.bfloat16, _attn_implementation="eager"
    ).to(args.DEVICE)
    model.eval()
    
    timing_log["1. Model Loading"] = time.perf_counter() - model_load_start
    print(f"Models loaded to {args.DEVICE} in {timing_log['1. Model Loading']:.2f} seconds.")

    # --- 2. Load and Ingest Data from .pkl File ---
    print(f"\n--- 2. Ingesting Data from {args.PKL_FILE_PATH} ---")
    ingestion_start = time.perf_counter()
    
    if not os.path.exists(args.PKL_FILE_PATH):
        print(f"[ERROR] PKL file not found at: {args.PKL_FILE_PATH}")
        return None # Return None on failure
        
    with open(args.PKL_FILE_PATH, "rb") as f:
        buffer: ObsDataBuffer = pickle.load(f)

    perception_log = PerceptionLog(args)

    
    total_pairs = 0 #counter
    #PROCESS_LIMIT = 150 # desired image limit - now just check first 150 images
    limit_reached = False 

    for entry in buffer.entries.values():
        if not entry.is_frame_full(): continue
        
        odom_to_base = {"position": entry.odometry["position"], "orientation": entry.odometry["orientation"]}
        
        for rgb_name, rgb_image, depth_name, depth_image in entry.get_rgb_depth_pairs():
            
            limit_is_active = isinstance(args.PROCESS_LIMIT, int) and args.PROCESS_LIMIT > 0
            if limit_is_active and total_pairs >= args.PROCESS_LIMIT:
                limit_reached = True
                break

            # sesize RGB to match depth
            if rgb_image.shape[:2] != depth_image.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))

            # map sensor name to transform name
            camera_mapping = {"head_rgb_left": "head_left_rgbd", "head_rgb_right": "head_right_rgbd", "left_rgb": "left_rgbd", "right_rgb": "right_rgbd", "rear_rgb": "rear_rgbd"}
            camera_link_name = camera_mapping.get(rgb_name)
            if not camera_link_name: continue

            # get world-to-camera transform
            w2c = compose_transforms_optimized(odom_to_base, camera_link_name, buffer.static_transforms, use_optical=True)
            
            # convert pose to the [w, x, y, z, px, py, pz] format 
            pos = w2c["position"]
            orient = w2c["orientation"]
            # the pkl buffer uses (x, y, z, w).
            quat_ros_wxyz = np.array([orient['w'], orient['x'], orient['y'], orient['z']])
            pos_xyz = np.array([pos['x'], pos['y'], pos['z']])
            camera_pose_7d = np.concatenate([quat_ros_wxyz, pos_xyz])

            # add data to our log
            perception_log.add_image(rgb_image, depth_image, camera_pose_7d)
            total_pairs += 1
            
        if limit_reached:
            break

    timing_log["2. Data Ingestion"] = time.perf_counter() - ingestion_start
    print(f"Ingested {total_pairs} RGB-Depth pairs in {timing_log['2. Data Ingestion']:.2f} seconds.")

    # --- 3. Run AI Processing Pipeline ---
    print("\n--- 3. Starting AI Processing Pipeline ---")
    processing_start_time = time.perf_counter()
    
    try:
        
        # --- De-duplication Log Setup ---
        log_filepath = os.path.join(perception_log.scan_dir, "deduplication_log.csv")
        log_file = open(log_filepath, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "image_id", "mask_index", "new_mask_centroid",
            "compared_to_object_id", "existing_object_centroid",
            "centroid_distance", "centroid_check_passed",
            "coverage_check_performed", "coverage_score_1", "coverage_score_2",
            "coverage_check_passed", "final_match_id"
        ])

        merge_counters = {}
        unique_object_point_clouds = {}
        per_frame_detections = {}
        best_instance_for_object = {}

        # CENTROID_DISTANCE_THRESHOLD = 200 #millimeters
        # COVERAGE_THRESHOLD = 0.70
        # VOXEL_SIZE = 0.30 #0.05 #= 50x50 mm

        for image_id, image_data in perception_log.data["images"].items():
            print(f"\n--- Processing {image_id} ---")
            
            sam_start = time.perf_counter()
            rgb_np = np.array(Image.open(image_data["rgb_path"]))
            depth_np = np.load(image_data["depth_path"])
            camera_pose = np.array(image_data["camera_pose_world"])
            
            masks = mask_generator.generate(rgb_np)
            sam_duration_ms = (time.perf_counter() - sam_start) * 1000
            timing_log["3a. SAM per image"].append({
                "image_id": image_id,
                "time_ms": round(sam_duration_ms, 3)
            })
            # record the number of masks found in this image
            timing_log["3e. Masks per image"].append({
                "image_id": image_id,
                "mask_count": len(masks)
            })
            print(f"  > SAM found {len(masks)} masks, (took {sam_duration_ms:.3f}ms).")

            h, w, _ = rgb_np.shape
            o3d_intrinsics = get_o3d_cam_intrinsic(h, w)
            organized_pcd = get_organized_point_cloud(depth_np, camera_pose, o3d_intrinsics)

            per_frame_detections[image_id] = []
            new_objects_in_this_frame = 0

            #initialize counters for dedup image's report
            centroid_checks_this_image = 0
            coverage_checks_this_image = 0
            total_centroid_time_ms = 0.0
            total_coverage_time_ms = 0.0

            for mask_index, mask in enumerate(masks):
                # Bounding Box Sanity Check: if it has 4 coordinates.
                bbox = mask.get('bbox')
                if bbox is None or len(bbox) != 4:
                    continue 
                
                object_points = organized_pcd[mask['segmentation']]
                if object_points.shape[0] < 50: continue

                new_centroid = np.mean(object_points, axis=0)
                matched_id = None

                # Loop through all known unique objects to check for a match
                for obj_id, data in unique_object_point_clouds.items():
                    # --- Stage 1: Centroid Distance Check ---
                    centroid_start = time.perf_counter()
                    existing_centroid = data['centroid']
                    dist = np.linalg.norm(new_centroid - existing_centroid)
                    centroid_duration_ms = (time.perf_counter() - centroid_start) * 1000
                    timing_log["3b. Dedup Centroid Check per mask"].append({
                        "image_id": image_id, "mask_index": mask_index,
                        "time_ms": round(centroid_duration_ms, 3)
                    })
                    
                    centroid_checks_this_image += 1
                    total_centroid_time_ms += centroid_duration_ms

                    centroid_passed = dist < args.centroid_threshold
                    coverage_performed, cov1, cov2, coverage_passed = False, 0.0, 0.0, False

                    # --- Stage 2: Coverage Check (only if centroid check passes) ---
                    if centroid_passed:
                        coverage_performed = True
                        coverage_start = time.perf_counter()
                        coverage1 = calculate_point_cloud_coverage(data['pcd'], object_points, voxel_size=args.voxel_size)
                        coverage2 = calculate_point_cloud_coverage(object_points, data['pcd'], voxel_size=args.voxel_size)
                        coverage_duration_ms = (time.perf_counter() - coverage_start) * 1000
                        timing_log["3c. Dedup Coverage Check per mask"].append({
                            "image_id": image_id, "mask_index": mask_index,
                            "time_ms": round(coverage_duration_ms, 3)
                        })
                        
                        coverage_checks_this_image += 1
                        total_coverage_time_ms += coverage_duration_ms

                        cov1, cov2 = round(coverage1, 4), round(coverage2, 4)
                        coverage_passed = cov1 > args.coverage_threshold or cov2 > args.coverage_threshold
                        if coverage_passed:
                            matched_id = obj_id
                    
                    # --- Write the detailed log row for this comparison ---
                    log_writer.writerow([
                        image_id, mask_index, np.round(new_centroid, 3),
                        obj_id, np.round(existing_centroid, 3),
                        round(dist, 4), centroid_passed,
                        coverage_performed, cov1, cov2,
                        coverage_passed, matched_id
                    ])

                    if matched_id:
                        break # Found a match, stop comparing (can be change later)

                is_new_object = matched_id is None
                if is_new_object:
                    image_index = int(image_id.split('_')[-1])
                    new_objects_in_this_frame += 1
                    object_id = f"obj_{image_index}_{new_objects_in_this_frame}"
                    unique_object_point_clouds[object_id] = {'pcd': object_points, 'centroid': new_centroid}
                    perception_log.add_or_update_unique_object(object_id, "[AWAITING VLM]", new_centroid)

                    # --- Source Instance Visualization ---
                    # a sub-folder for this new unique object
                    merge_folder_path = os.path.join(perception_log.scan_dir, "merge_tracking", object_id)
                    os.makedirs(merge_folder_path, exist_ok=True)
                    # save the visualization for this first instance
                    #x, y, w, h = mask['bbox']
                    x, y, w, h = [int(c) for c in mask['bbox']] # Convert coordinates to integers
                    cropped_image = rgb_np[y:y+h, x:x+w]
                    instance_label = f"Source: {image_id}_mask_{mask_index}"
                    instance_viz = create_object_visualization(cropped_image, instance_label)
                    instance_path = os.path.join(merge_folder_path, f"{instance_label}.png")
                    Image.fromarray(instance_viz).save(instance_path)

                else:
                    object_id = matched_id
                    merged_pcd = np.vstack((unique_object_point_clouds[object_id]['pcd'], object_points))
                    new_avg_centroid = np.mean(merged_pcd, axis=0)
                    unique_object_point_clouds[object_id] = {'pcd': merged_pcd, 'centroid': new_avg_centroid}

                    # --- Add Source Instance Visualization ---
                    # get the path to the existing object's folder
                    merge_folder_path = os.path.join(perception_log.scan_dir, "merge_tracking", object_id)
                    # create and save the visualization for THIS duplicate instance
                    #x, y, w, h = mask['bbox']
                    x, y, w, h = [int(c) for c in mask['bbox']] # Convert coordinates to integers
                    cropped_image = rgb_np[y:y+h, x:x+w]
                    instance_label = f"Source: {image_id}_mask_{mask_index}"
                    instance_viz = create_object_visualization(cropped_image, instance_label)
                    instance_path = os.path.join(merge_folder_path, f"{instance_label}.png")
                    Image.fromarray(instance_viz).save(instance_path)

                per_frame_detections[image_id].append({"object_id": object_id, "is_new": is_new_object, "mask": mask['segmentation'], "bbox": mask['bbox']})
                current_instance_info = {"mask_area": mask['area'], "image_id": image_id, "bbox": mask['bbox']}
                if object_id not in best_instance_for_object or mask['area'] > best_instance_for_object[object_id]['mask_area']:
                    best_instance_for_object[object_id] = current_instance_info
                perception_log.add_object_instance(image_id, object_id, mask['bbox'], mask['area'], mask['segmentation'])

            # --- Print the detailed summary for the completed image ---
            print(f"  > Dedup Centroid: {centroid_checks_this_image} checks took {total_centroid_time_ms:.3f}ms")
            if coverage_checks_this_image > 0:
                print(f"  > Dedup Coverage: {coverage_checks_this_image} checks took {total_coverage_time_ms:.3f}ms")
        
        
        print("\n--- Generating VLM Descriptions for Unique Objects ---")
        for object_id, data in unique_object_point_clouds.items():
            if object_id not in best_instance_for_object: continue
            
            vlm_start = time.perf_counter()
            avg_position = data['centroid']
            best_instance_data = best_instance_for_object[object_id]
            parent_image_path = perception_log.data["images"][best_instance_data["image_id"]]["rgb_path"]
            rgb_np = np.array(Image.open(parent_image_path))
            #x, y, w, h = best_instance_data["bbox"]
            x, y, w, h = [int(c) for c in best_instance_data["bbox"]] 

            pad_x, pad_y = int(w * args.vlm_padding), int(h * args.vlm_padding)
            img_h, img_w, _ = rgb_np.shape
            x1_padded, y1_padded = max(0, x - pad_x), max(0, y - pad_y)
            x2_padded, y2_padded = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
            cropped_image = rgb_np[y1_padded:y2_padded, x1_padded:x2_padded]
            
            description = get_vlm_description(cropped_image, model, processor)
            vlm_duration_ms = (time.perf_counter() - vlm_start) * 1000
            timing_log["3d. VLM Description per object"].append({
                "object_id": object_id,
                "time_ms": round(vlm_duration_ms, 3)
            })
            print(f"  > {object_id}: '{description}' (took {vlm_duration_ms:.3f}ms).")
            
            perception_log.add_or_update_unique_object(object_id, description, avg_position)

             # ---  Object Frame Visualization ---
            # shows the unique object in its original full-frame context
            best_instance_info = best_instance_for_object[object_id]
            parent_image_id = best_instance_info["image_id"]
            parent_image_path = perception_log.data["images"][parent_image_id]["rgb_path"]
            rgb_for_viz = np.array(Image.open(parent_image_path))
            
            # find the instance_id to load the correct mask
            for inst_id, inst_data in perception_log.data["object_instances"].items():
                if inst_data["parent_image_id"] == parent_image_id and inst_data["bounding_box"] == best_instance_info["bbox"]:
                    mask_np = np.load(inst_data["mask_path"])
                    viz_image = create_visualization(
                        rgb_for_viz, mask_np, best_instance_info["bbox"], f"{object_id}: {description}"
                    )
                    viz_path = os.path.join(perception_log.scan_dir, "obj_frame_visualizations", f"{object_id}.png")
                    Image.fromarray(viz_image).save(viz_path)
                    perception_log.add_visualization_path_to_object(object_id, viz_path)
                    break # stop after finding the matching instance (could be modify later)

        log_file.close() # close the log file after processing is done
    
    except Exception as e:
        print(f"\n--- ERROR DURING AI PROCESSING: {e} ---")
    finally:
        timing_log["3. Full AI Processing"] = time.perf_counter() - processing_start_time
        sam.to('cpu'); model.to('cpu'); torch.cuda.empty_cache()
        print("Moved models to CPU to free VRAM.")

    # --- 4. Generate Final Visualizations ---
    print("\n--- 4. Generating Final Visualizations ---")
    viz_start = time.perf_counter()

    # object-specific visualizations
    for object_id, best_instance_data in best_instance_for_object.items():
        parent_image_path = perception_log.data["images"][best_instance_data["image_id"]]["rgb_path"]
        rgb_np = np.array(Image.open(parent_image_path))
        #x, y, w, h = best_instance_data["bbox"]
        x, y, w, h = [int(c) for c in best_instance_data["bbox"]]        
        cropped_image = rgb_np[y:y+h, x:x+w]
        desc = perception_log.data["unique_objects"][object_id]["vlm_description"]
        obj_label = f"{object_id}: {desc}"
        object_viz_image = create_object_visualization(cropped_image, obj_label)
        obj_viz_path = os.path.join(perception_log.scan_dir, "object_visualization", f"{object_id}.png")
        Image.fromarray(object_viz_image).save(obj_viz_path)
    
    # per-frame visualizations
    for image_id, detections in per_frame_detections.items():
        frame_rgb_path = perception_log.data["images"][image_id]["rgb_path"]
        frame_viz_image = np.array(Image.open(frame_rgb_path))
        for det in detections:
            obj_id = det["object_id"]
            desc = perception_log.data["unique_objects"][obj_id]["vlm_description"]
            label, color = f"{obj_id}: {desc}", (0, 255, 0) if det["is_new"] else (255, 0, 0)
            draw_detection_on_image(frame_viz_image, det["mask"], det["bbox"], label, color)
        frame_viz_path = os.path.join(perception_log.scan_dir, "frame_visualizations", f"{image_id}.png")
        Image.fromarray(frame_viz_image).save(frame_viz_path)
    
    timing_log["4. Visualization"] = time.perf_counter() - viz_start
    print(f"All visualizations saved in {timing_log['4. Visualization']:.2f} seconds.")

    # finalize log and prepare for return
    perception_log.finalize_and_save(source_file=args.PKL_FILE_PATH)
    timing_log["Total Time"] = time.perf_counter() - overall_start_time
    perception_log.timing_log = timing_log
    return perception_log

    # --- 5. Finalize, Analyze, and Report ---


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the offline perception pipeline on a pre-recorded obs_buffer.pkl file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    # --- Core Configuration Arguments ---
    parser.add_argument('-p', '--PKL_FILE_PATH', type=str, required=True, 
                        help='Required: Path to the input obs_buffer.pkl file.')
    parser.add_argument('-s', '--SAM_CHECKPOINT_PATH', type=str, required=True, 
                        help='Required: Path to the SAM checkpoint model file (.pth).')
    parser.add_argument('-v', '--VLM_MODEL_ID', type=str, default='HuggingFaceTB/SmolVLM-256M-Instruct',
                        help='Hugging Face model ID for the Vision-Language Model.')
    parser.add_argument('-d', '--DEVICE', type=str, default=None,
                        help='Device to use ("cuda" or "cpu"). If not set, it will auto-detect CUDA.')
    parser.add_argument('-l', '--PROCESS_LIMIT', type=int, default=150,
                        help='Maximum number of images to process. Set to 0 to process all images.')

    # --- Algorithm Tuning Arguments ---
    parser.add_argument('--centroid_threshold', type=int, default=200,
                        help='De-duplication: Maximum distance in millimeters between centroids for a potential match.')
    parser.add_argument('--coverage_threshold', type=float, default=0.70,
                        help='De-duplication: Minimum point cloud overlap percentage to confirm a match.')
    parser.add_argument('--voxel_size', type=float, default=0.30,
                        help='De-duplication: Voxel size in meters for the point cloud coverage check.')
    parser.add_argument('--min_mask_area', type=int, default=100,
                        help='SAM: Minimum area in pixels for a mask to be considered a valid object.')
    parser.add_argument('--vlm_padding', type=float, default=0.15,
                        help='VLM: Percentage of padding to add around an object before sending to the VLM.')
    
    args = parser.parse_args()

    # --- Post-parsing logic ---
    # Handle auto-detection for DEVICE if it's not specified by the user
    if args.DEVICE is None:
        args.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Execute the main pipeline ---
    perception_log = main(args)

    # --- Analyze results ---
    if perception_log:
        timing_log = perception_log.timing_log
        # Pass the process_limit from the parsed arguments
        analyze_and_save_timing_log(perception_log, timing_log, args.PROCESS_LIMIT) 
        print(f"\nProcessing complete. Results are in: {perception_log.scan_dir}")
    
