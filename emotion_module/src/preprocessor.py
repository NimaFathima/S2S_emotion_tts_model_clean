# src/preprocessor.py
import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from config.settings import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

def apply_clahe_channels(
    bgr_frame: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID_SIZE,
    exposure_threshold: float = 0.15  # Max 15% of pixels can be completely blown out
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Preprocesses frames and returns a tuple: (processed_frame, is_overexposed)
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None, False

    try:
        lab_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_frame)

        # Check for overexposure: Calculate percentage of pixels near max intensity (255)
        # In LAB, L ranges from 0 (black) to 255 (white) in OpenCV implementation
        blown_out_pixels = np.sum(l_channel >= 250)
        total_pixels = l_channel.size
        overexposed_ratio = blown_out_pixels / total_pixels

        is_overexposed = bool(overexposed_ratio > exposure_threshold)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl_channel = clahe.apply(l_channel)

        merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR), is_overexposed

    except (cv2.error, OSError, ValueError) as e:
        logging.error(f"Preprocessor failure: {e}", exc_info=True)
        return None, False
