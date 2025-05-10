import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
# Add imports for video processing
import cv2
import math
from PIL import Image
import torchvision.transforms as transforms
# Add subprocess for ffprobe
import subprocess
import json

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        print(text)
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss

# def strip_lowerdiag(L):
#     if L.shape[1] == 3:
#         uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
#         uncertainty[:, 0] = L[:, 0, 0]
#         uncertainty[:, 1] = L[:, 0, 1]
#         uncertainty[:, 2] = L[:, 0, 2]
#         uncertainty[:, 3] = L[:, 1, 1]
#         uncertainty[:, 4] = L[:, 1, 2]
#         uncertainty[:, 5] = L[:, 2, 2]

#     elif L.shape[1] == 2:
#         uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
#         uncertainty[:, 0] = L[:, 0, 0]
#         uncertainty[:, 1] = L[:, 0, 1]
#         uncertainty[:, 2] = L[:, 1, 1]
#     return uncertainty

# def strip_symmetric(sym):
#     return strip_lowerdiag(sym)

# def build_rotation(r):
#     norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

#     q = r / norm[:, None]

#     R = torch.zeros((q.size(0), 3, 3), device='cuda')

#     r = q[:, 0]
#     x = q[:, 1]
#     y = q[:, 2]
#     z = q[:, 3]

#     R[:, 0, 0] = 1 - 2 * (y*y + z*z)
#     R[:, 0, 1] = 2 * (x*y - r*z)
#     R[:, 0, 2] = 2 * (x*z + r*y)
#     R[:, 1, 0] = 2 * (x*y + r*z)
#     R[:, 1, 1] = 1 - 2 * (x*x + z*z)
#     R[:, 1, 2] = 2 * (y*z - r*x)
#     R[:, 2, 0] = 2 * (x*z - r*y)
#     R[:, 2, 1] = 2 * (y*z + r*x)
#     R[:, 2, 2] = 1 - 2 * (x*x + y*y)
#     return R

# def build_scaling_rotation(s, r):
#     L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
#     R = build_rotation(r)

#     L[:,0,0] = s[:,0]
#     L[:,1,1] = s[:,1]
#     L[:,2,2] = s[:,2]

#     L = R @ L
#     return L

# def build_rotation_2d(r):
#     '''
#     Build rotation matrix in 2D.
#     '''
#     R = torch.zeros((r.size(0), 2, 2), device='cuda')
#     R[:, 0, 0] = torch.cos(r)[:, 0]
#     R[:, 0, 1] = -torch.sin(r)[:, 0]
#     R[:, 1, 0] = torch.sin(r)[:, 0]
#     R[:, 1, 1] = torch.cos(r)[:, 0]
#     return R

# def build_scaling_rotation_2d(s, r, device):
#     L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
#     R = build_rotation_2d(r, device)
#     L[:,0,0] = s[:,0]
#     L[:,1,1] = s[:,1]
#     L = R @ L
#     return L

# def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
#     '''
#     Build covariance metrix from rotation and scale matricies.
#     '''
#     L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
#     actual_covariance = L @ L.transpose(1, 2)
#     return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R

def video_path_to_tensor(video_path, num_frames=None, target_size=None, target_pixel_count=65536):
    """Reads frames from a video file, resizes them, and returns a tensor.

    Args:
        video_path (str): Path to the video file.
        num_frames (int, optional): Number of frames to read. Reads all if None.
        target_size (tuple, optional): Target size (width, height). If None,
                                     resizes based on a heuristic similar to
                                     the original image processing.
        target_pixel_count (int, optional): Target number of pixels for resizing.
                                          Default is 65536 (256*256).

    Returns:
        torch.Tensor: Tensor containing video frames of shape (T, C, H, W).
    """
    # Get rotation and accurate frame count using ffprobe
    rotation_angle, ffprobe_total_frames = get_video_properties_ffprobe(video_path)

    if ffprobe_total_frames is None:
        # Fallback to OpenCV if ffprobe fails for frame count, though it might be inaccurate
        print("Warning: ffprobe failed to get frame count. Falling back to OpenCV for frame count (may be inaccurate).")
        cap_for_count = cv2.VideoCapture(str(video_path))
        if not cap_for_count.isOpened():
            raise IOError(f"Cannot open video file for count: {video_path}")
        opencv_total_frames = int(cap_for_count.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_for_count.release()
        total_frames = opencv_total_frames
        if total_frames <= 0: # Further check if OpenCV also fails badly
            raise ValueError(f"Could not determine a valid frame count for {video_path}")
    else:
        total_frames = ffprobe_total_frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get OpenCV reported FPS for reference, but total_frames is now from ffprobe (more reliable)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Properties: Total Frames (reliable) = {total_frames}, OpenCV FPS = {video_fps:.2f}")

    frames_to_capture = total_frames
    selected_indices = None

    if num_frames is not None and num_frames < total_frames:
        print(f"Selecting {num_frames} frames evenly spaced from {total_frames} total frames.")
        # Generate num_frames evenly spaced indices
        # Ensure indices are within the valid range [0, total_frames - 1]
        indices = np.floor(np.linspace(0, total_frames -1 - 1e-9, num_frames)).astype(int)
        # An alternative, often more robust method for selecting N frames from V:
        # indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        # However, linspace (with safety for endpoint) should also work well if total_frames is accurate.
        # To be extra safe, clip values just in case of any float precision issues:
        indices = np.clip(indices, 0, total_frames - 1)

        selected_indices = set(indices) # Use set to ensure uniqueness if any rounding produced duplicates
        frames_to_capture = len(selected_indices)

        if frames_to_capture < num_frames:
            print(f"Warning: Expected to select {num_frames} frames, but only got {frames_to_capture} unique indices. This might be due to rounding or video length.")
        print(f"Selected frame indices: {sorted(list(selected_indices))}")
    elif num_frames is not None:
        print(f"Requested {num_frames} frames, but video only has {total_frames}. Using all frames.")
        # If user requested more frames than available, just use all of them.
        num_frames = total_frames # Update num_frames to actual count
        # No need to select indices, will capture all
    else:
        # num_frames was None, use all frames
        num_frames = total_frames
        print(f"Processing all {total_frames} frames.")
        # No need to select indices

    frames = []
    processed_frame_count = 0
    current_frame_index = 0
    original_w, original_h = None, None

    while processed_frame_count < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {current_frame_index} or end of video reached early.")
            break

        # Check if we need to select this frame
        should_process_frame = (selected_indices is None) or (current_frame_index in selected_indices)

        if should_process_frame:

            # --- Apply rotation if needed ---
            if rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # --- End rotation ---

            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            if processed_frame_count == 0: # First frame *being processed*
                original_w, original_h = img.size
                print(f"Original Frame size: ({original_w}, {original_h})")
                if target_size is None:
                    # Heuristic resizing to aim for target_pixel_count pixels
                    if original_w * original_h > 0: # Check to avoid division by zero
                        if original_w * original_h > target_pixel_count:
                            # Only scale down if the original is larger than the target
                            scale = math.sqrt(target_pixel_count / (original_w * original_h))
                        else:
                            # If original is smaller or equal, do not scale up
                            scale = 1.0
                    else:
                        scale = 1.0 # Fallback scale if original dimensions are invalid

                    target_w = max(16, int(round(original_w * scale / 16)) * 16)
                    target_h = max(16, int(round(original_h * scale / 16)) * 16)
                    current_target_size = (target_w, target_h)
                    print(f"Resizing selected frames to: {current_target_size} (Original: {original_w}x{original_h}, Target Pixels: ~{target_pixel_count}, Scale: {scale:.4f})")
                else:
                    current_target_size = target_size
                    print(f"Using provided target size: {current_target_size}")

            # Resize image
            img_resized = img.resize(current_target_size)

            # Convert to tensor
            transform = transforms.ToTensor() # Scales to [0, 1]
            img_tensor = transform(img_resized) # Shape (C, H, W)
            frames.append(img_tensor)
            processed_frame_count += 1

        current_frame_index += 1

    cap.release()

    if not frames:
        raise ValueError(f"Could not read any frames from {video_path} (or no frames selected)")

    # Stack frames into a single tensor (T, C, H, W)
    video_tensor = torch.stack(frames)
    print(f"Final video tensor shape: {video_tensor.shape}") # Should match frames_to_capture

    # Calculate effective output FPS for duration preservation
    effective_output_fps = 25.0 # Default fallback
    if total_frames > 0 and frames_to_capture > 0 and video_fps is not None and video_fps > 0:
        effective_output_fps = video_fps * (frames_to_capture / total_frames)
        print(f"Calculated effective output FPS for duration preservation: {effective_output_fps:.2f} (Original FPS: {video_fps:.2f}, Frames Captured: {frames_to_capture}, Total Original: {total_frames})")
    elif video_fps is not None and video_fps > 0:
        effective_output_fps = video_fps # Fallback to original FPS if frame counts are problematic for ratio
        print(f"Warning: Using original video FPS ({video_fps:.2f}) as effective FPS could not be calculated reliably due to frame counts.")
    else:
        print(f"Warning: Could not determine valid original video FPS. Using default effective FPS: {effective_output_fps:.2f} FPS.")

    return video_tensor, effective_output_fps

def get_video_properties_ffprobe(video_path):
    """Uses ffprobe to get rotation and accurate frame count from a video file.

    Returns:
        tuple: (rotation_angle, frame_count)
               rotation_angle is int (0, 90, 180, 270) or 0 if not found/error.
               frame_count is int or None if error.
    """
    rotation_angle = 0
    frame_count = None
    try:
        # Get rotation
        cmd_rotate = [
            'ffprobe',
            '-loglevel', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate',
            '-of', 'default=nw=1:nk=1',
            str(video_path)
        ]
        result_rotate = subprocess.run(cmd_rotate, capture_output=True, text=True, check=False) # check=False to handle missing tag gracefully
        if result_rotate.returncode == 0 and result_rotate.stdout.strip():
            try:
                angle = int(result_rotate.stdout.strip())
                if angle in [0, 90, 180, 270]:
                    rotation_angle = angle
                    print(f"Detected video rotation (ffprobe): {rotation_angle} degrees")
                else:
                    print(f"Warning: Unexpected rotation value {angle} from ffprobe. Assuming 0.")
            except ValueError:
                print("Warning: Could not parse rotation value from ffprobe. Assuming 0.")
        else:
            print("Info: No rotation tag found or ffprobe error for rotation. Assuming 0 rotation.")

        # Get frame count
        cmd_frames = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'default=nokey=1:noprint_wrappers=1',
            str(video_path)
        ]
        result_frames = subprocess.run(cmd_frames, capture_output=True, text=True, check=True)
        frame_count = int(result_frames.stdout.strip())
        print(f"Detected frame count (ffprobe): {frame_count}")

    except FileNotFoundError:
        print("ERROR: ffprobe command not found. Cannot get accurate video properties.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffprobe failed to get frame count. Error: {e}")
    except ValueError:
        print("ERROR: Could not parse frame count from ffprobe.")

    return rotation_angle, frame_count
