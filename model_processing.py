import os
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

# --- Global variable to hold the model ---
# This prevents reloading the model on every request, which is very slow.
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_weights_path):
    """
    Loads the segmentation model into a global variable.
    This function is called once when the Flask app starts.
    It will raise a RuntimeError if loading fails.
    """
    global MODEL
    if MODEL is None:
        print(f"Loading model from {model_weights_path} onto {DEVICE}...")
        try:
            # Check if the model file exists before trying to load it
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(f"Model weights file not found at: {model_weights_path}")

            MODEL = smp.UnetPlusPlus(
                encoder_name="efficientnet-b1",
                encoder_weights=None,
                in_channels=5,
                classes=1,
            ).to(DEVICE)
            MODEL.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
            MODEL.eval()  # Set model to evaluation mode
            print("✅ Model loaded successfully.")
        except Exception as e:
            # Raise the exception to halt the application startup
            MODEL = None
            raise RuntimeError(
                f"❌ CRITICAL ERROR: Could not load model. Please check the path and file integrity. Original error: {e}")

def _resize_nifti(input_path, output_path, target_shape=(192, 192)):
    """
    Internal function to resize a NIfTI image.
    (This is a slightly modified version of your original function)
    """
    try:
        nii = nib.load(input_path)
        data = nii.get_fdata()

        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]

        if data.ndim != 3:
            raise ValueError(f"Input image must be 3D, but got shape {data.shape}")

        zoom_factors = [target_shape[0] / data.shape[0], target_shape[1] / data.shape[1], 1.0]
        resized_data = zoom(data, zoom_factors, order=1, prefilter=False)

        # Use original affine and header to preserve orientation and metadata
        new_nii = nib.Nifti1Image(resized_data.astype(np.float32), affine=nii.affine, header=nii.header)
        nib.save(new_nii, output_path)
        return True
    except Exception as e:
        print(f"❌ ERROR resizing {os.path.basename(input_path)}: {e}")
        return False

def _segment_nifti(input_path, output_path, progress_callback=None):
    """
    Internal function to apply segmentation to a preprocessed NIfTI file.
    Accepts an optional progress_callback(percent:int) for reporting progress.
    """
    if MODEL is None:
        raise RuntimeError("Model is not loaded. Please call load_model() first.")

    try:
        nifti_img = nib.load(input_path)
        img_3d_data = nifti_img.get_fdata().astype(np.float32)
        original_affine = nifti_img.affine
        original_header = nifti_img.header
        MODEL_INPUT_SIZE = (192, 192)
        num_slices = img_3d_data.shape[2]

        min_val, max_val = np.min(img_3d_data), np.max(img_3d_data)
        img_3d_normalized = (img_3d_data - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-8 else np.zeros_like(img_3d_data)

        output_slices = []
        for slice_idx in range(num_slices):
            slice_stack = []
            for i in range(-2, 3):
                context_slice_idx = slice_idx + i
                if 0 <= context_slice_idx < num_slices:
                    slice_stack.append(img_3d_normalized[:, :, context_slice_idx])
                else:
                    slice_stack.append(np.zeros_like(img_3d_normalized[:, :, 0]))

            img_multichannel = np.stack(slice_stack, axis=0)
            img_tensor = torch.tensor(img_multichannel, dtype=torch.float32)

            img_tensor_resized = TF.resize(img_tensor, MODEL_INPUT_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
            input_tensor = img_tensor_resized.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output_logits = MODEL(input_tensor)

            pred_mask = (torch.sigmoid(output_logits) > 0.5).float()
            pred_mask_resized = TF.resize(pred_mask, (img_3d_data.shape[0], img_3d_data.shape[1]), interpolation=InterpolationMode.NEAREST)
            output_slice_np = pred_mask_resized.squeeze().cpu().numpy()
            output_slices.append(output_slice_np)

            # --- Progress update: 50% to 80% ---
            if progress_callback:
                progress = 50 + int(30 * (slice_idx + 1) / num_slices)
                progress_callback(progress)

        output_volume = np.stack(output_slices, axis=2)
        output_nifti = nib.Nifti1Image(output_volume.astype(np.uint8), original_affine, original_header)
        nib.save(output_nifti, output_path)
        return True
    except Exception as e:
        print(f"❌ ERROR during segmentation: {e}")
        return False

def _resize_mask_to_original(mask_path, output_path, original_shape, original_affine, original_header):
    """
    Resizes the mask NIfTI file at mask_path back to the original_shape and saves to output_path.
    Uses nearest-neighbor interpolation to preserve mask values.
    """
    try:
        nii = nib.load(mask_path)
        data = nii.get_fdata()
        # Only resize spatial dimensions (x, y), keep z the same
        zoom_factors = [original_shape[0] / data.shape[0], original_shape[1] / data.shape[1], original_shape[2] / data.shape[2]]
        resized_data = zoom(data, zoom_factors, order=0, prefilter=False)
        new_nii = nib.Nifti1Image(resized_data.astype(np.uint8), affine=original_affine, header=original_header)
        nib.save(new_nii, output_path)
        return True
    except Exception as e:
        print(f"❌ ERROR resizing mask to original shape: {e}")
        return False

def run_segmentation_pipeline(input_filepath, output_filepath, progress_callback=None):
    """
    Orchestrates the full preprocessing and segmentation pipeline.
    Accepts an optional progress_callback(percent:int) for reporting progress.
    """
    temp_dir = os.path.dirname(input_filepath)
    filename = os.path.basename(input_filepath)
    intermediate_resized_path = os.path.join(temp_dir, f"resized_{filename}")
    intermediate_mask_path = os.path.join(temp_dir, f"mask_resized_{filename}")
    orig_nii = nib.load(input_filepath)
    original_shape = orig_nii.get_fdata().shape
    original_affine = orig_nii.affine
    original_header = orig_nii.header
    if progress_callback: progress_callback(20)
    if not _resize_nifti(input_filepath, intermediate_resized_path):
        if progress_callback: progress_callback(-1)
        return False, "Failed during the resizing step."
    # --- Pass progress_callback to segmentation step ---
    if not _segment_nifti(intermediate_resized_path, intermediate_mask_path, progress_callback=progress_callback):
        if os.path.exists(intermediate_resized_path):
            os.remove(intermediate_resized_path)
        if progress_callback: progress_callback(-1)
        return False, "Failed during the segmentation step."
    if progress_callback: progress_callback(80)
    if not _resize_mask_to_original(intermediate_mask_path, output_filepath, original_shape, original_affine, original_header):
        if os.path.exists(intermediate_resized_path):
            os.remove(intermediate_resized_path)
        if os.path.exists(intermediate_mask_path):
            os.remove(intermediate_mask_path)
        if progress_callback: progress_callback(-1)
        return False, "Failed during the mask resizing step."
    if os.path.exists(intermediate_resized_path):
        os.remove(intermediate_resized_path)
    if os.path.exists(intermediate_mask_path):
        os.remove(intermediate_mask_path)
    if progress_callback: progress_callback(100)

    return True, "Segmentation successful."
