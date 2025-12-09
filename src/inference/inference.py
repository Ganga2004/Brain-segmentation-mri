#!/usr/bin/env python3
"""
Simple PyTorch inference script to load a .pt checkpoint and produce a segmentation mask.

Usage examples:
    python inference.py --model-path ./model.pt --input /path/to/slice.png --output ./out_mask.png --input-type png
    python inference.py --model-path ./checkpoint.pt --input /path/to/volume.nii.gz --output ./mask_out.nii.gz --input-type nii

If you have a model class defined in e.g. src/model.py, edit the MODEL IMPORT section below.
"""
import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Optional: nibabel for NIfTI support
try:
    import nibabel as nib
except Exception:
    nib = None

# ---------------------------
# EDIT THIS if you have a model class
# ---------------------------
# If you have your model implementation (for example HCSAF or UNet) in a file, import it here:
# from src.model import MyModel   # <-- replace with your model definition
#
# Example:
# from src.hcsaf_model import HCSAF
# MODEL_CLASS = HCSAF
#
# If you do NOT have the model class available, the loader will attempt to torch.load() the entire object.
MODEL_CLASS = None  # <-- set to your class if available, e.g. MODEL_CLASS = MyModel
MODEL_CONSTRUCTOR_KWARGS = {}  # fill if your model class requires args e.g. {"in_ch":1, "out_ch":1}
# ---------------------------

def load_model(model_path, device):
    """Tries to load model from checkpoint_path. Supports:
       - state_dict (when MODEL_CLASS != None)
       - full model saved with torch.save(model)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    if MODEL_CLASS is not None:
        model = MODEL_CLASS(**MODEL_CONSTRUCTOR_KWARGS)
        ckpt = torch.load(str(model_path), map_location=device)
        # ckpt might be a dict with 'state_dict' or may be the state_dict itself
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        # if keys are prefixed (like module.), try to strip
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # try stripping "module." prefixes
            new_state = {}
            for k, v in state.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_k] = v
            model.load_state_dict(new_state)
        model.to(device)
        model.eval()
        return model

    else:
        # Try to load whole model object (less recommended but possible)
        print("No MODEL_CLASS provided. Attempting to load full model object from checkpoint...")
        model = torch.load(str(model_path), map_location=device)
        # If loaded object is a dict containing state or model, attempt to extract
        if isinstance(model, dict):
            # common patterns:
            for candidate in ["model", "net", "state_dict", "state"]:
                if candidate in model:
                    model = model[candidate]
                    break
        # If it's state_dict still, user needs to set MODEL_CLASS.
        if isinstance(model, dict):
            raise RuntimeError("Loaded checkpoint is a state_dict. You must set MODEL_CLASS in the script to load state_dict.")
        model.to(device)
        model.eval()
        return model


def preprocess_image_pil(img_path, target_size=256, to_grayscale=True):
    img = Image.open(img_path)
    if to_grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    # resize (maintain aspect by center-crop then resize if needed)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    if to_grayscale:
        arr = arr[np.newaxis, ...]  # 1, H, W
    else:
        arr = arr.transpose(2, 0, 1)  # C, H, W
    return arr


def preprocess_nifti(nii_path, slice_index=None, target_size=256):
    if nib is None:
        raise RuntimeError("nibabel is required to read NIfTI files. Install via `pip install nibabel`.")
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    # Assuming shape (H, W, D) or (D, H, W) — try to find the largest 2D plane
    if data.ndim == 3:
        # choose middle slice if not provided
        if slice_index is None:
            slice_index = data.shape[2] // 2
        slice_2d = data[:, :, slice_index]
    else:
        raise ValueError("Unsupported NIfTI shape: " + str(data.shape))
    # normalize (min-max)
    sl = slice_2d.astype(np.float32)
    mn, mx = sl.min(), sl.max()
    if mx > mn:
        sl = (sl - mn) / (mx - mn)
    else:
        sl = sl - mn
    # resize to (target_size, target_size)
    pil = Image.fromarray((sl * 255).astype(np.uint8))
    pil = pil.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]  # 1, H, W
    return arr, img.affine if hasattr(img, 'affine') else None, data.shape


def postprocess_and_save(mask_prob, out_path, threshold=0.5, as_nifti=None, original_shape=None, affine=None):
    """
    mask_prob: numpy array HxW or 1xHxW
    out_path: file path
    if as_nifti=True: original_shape and affine required to upsample back (best-effort).
    """
    if mask_prob.ndim == 3:
        mask_prob = mask_prob[0]
    mask = (mask_prob >= threshold).astype(np.uint8) * 255
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save PNG
    if out_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        Image.fromarray(mask).save(str(out_path))
        print(f"Saved mask PNG to {out_path}")
    elif out_path.suffix.lower() in ['.nii', '.gz', '.nii.gz']:
        if nib is None:
            raise RuntimeError("To save NIfTI output, install nibabel (`pip install nibabel`).")
        if affine is None:
            affine = np.eye(4)
        # attempt to reshape to original shape if provided
        if original_shape is not None:
            # Simple nearest-neighbour upsample: use PIL to resize back to original shape HxW
            pil = Image.fromarray(mask)
            # assume original_shape is (H, W, D) and we only have a single slice — place it at middle slice
            H, W = original_shape[0], original_shape[1]
            pil = pil.resize((W, H), Image.NEAREST)
            arr = np.zeros(original_shape, dtype=np.uint8)
            mid = original_shape[2] // 2
            arr[:, :, mid] = np.array(pil)
            nii = nib.Nifti1Image(arr, affine)
            nib.save(nii, str(out_path))
            print(f"Saved mask NIfTI to {out_path}")
        else:
            nii = nib.Nifti1Image(mask.astype(np.uint8), affine if affine is not None else np.eye(4))
            nib.save(nii, str(out_path))
            print(f"Saved mask NIfTI to {out_path}")
    else:
        # default to png
        p = out_path.with_suffix(".png")
        Image.fromarray(mask).save(str(p))
        print(f"Saved mask PNG to {p}")


def run_inference(args):
    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    model = load_model(args.model_path, device)

    # read input
    if args.input_type == "png" or args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        arr = preprocess_image_pil(args.input, target_size=args.size, to_grayscale=True)
        original_shape = None
        affine = None
    elif args.input_type == "npz":
        data = np.load(args.input)
        # assume array named 'arr' or the first array
        if 'arr' in data:
            arr = data['arr'].astype(np.float32)
        else:
            arr = data[list(data.files)[0]].astype(np.float32)
        # ensure C,H,W
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        original_shape = None
        affine = None
    elif args.input_type == "nii":
        arr, affine, original_shape = preprocess_nifti(args.input, slice_index=args.slice_idx, target_size=args.size)
    else:
        raise ValueError("Unsupported input_type. Use png / npz / nii")

    # To torch tensor
    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)  # shape 1,C,H,W
    with torch.no_grad():
        out = model(x)
        # model output could be logits or probabilities
        if isinstance(out, (tuple, list)):
            out = out[0]
        # If out has channels e.g. (1,1,H,W) -> squeeze
        if out.ndim == 4:
            out = out.squeeze(0)  # 1,C,H,W -> C,H,W
        # if single-channel output
        if out.shape[0] > 1:
            # If multi-class, take argmax then one-hot for a particular class (0 = background)
            out_prob = torch.softmax(out, dim=0)[1:2, ...] if out.shape[0] > 1 else torch.sigmoid(out)
            # pick channel 1 if binary segmentation used channel 1 as foreground. If unsure, take channel 0 as prob.
            out_prob = out_prob.squeeze(0).cpu().numpy()
        else:
            out_prob = torch.sigmoid(out).cpu().numpy()
    # if out_prob shape C,H,W or H,W
    if out_prob.ndim == 3 and out_prob.shape[0] == 1:
        out_prob = out_prob[0]
    postprocess_and_save(out_prob, args.output, threshold=args.threshold, as_nifti=(args.input_type == 'nii'), original_shape=original_shape, affine=affine)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Path to .pt checkpoint")
    p.add_argument("--input", required=True, help="Path to input image (png, jpg) or .nii.gz or .npz")
    p.add_argument("--input-type", choices=["png", "npz", "nii"], default=None, help="Input type; if omitted, guessed from extension")
    p.add_argument("--output", required=True, help="Output path for mask (png or .nii.gz)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to run inference on")
    p.add_argument("--size", type=int, default=256, help="Resize input to this size (model trained on 256)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting prob->binary")
    p.add_argument("--slice-idx", type=int, default=None, help="For nifti input: slice index to extract (defaults to middle slice)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # infer input type if not provided
    if args.input_type is None:
        ext = os.path.splitext(args.input)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            args.input_type = "png"
        elif ext in [".npz"]:
            args.input_type = "npz"
        elif ext in [".nii", ".gz", ".nii.gz"]:
            args.input_type = "nii"
        else:
            raise ValueError("Cannot guess input type from extension. Use --input-type to specify.")
    run_inference(args)
