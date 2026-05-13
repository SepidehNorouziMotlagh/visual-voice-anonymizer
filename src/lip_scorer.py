"""
lip_scorer.py
=============
Computes a per‑frame lip‑activity score (0‑1) from a video using
MediaPipe Face Landmarker (Tasks API).

Requirements (already in requirements.txt):
  - mediapipe >= 0.10.0
  - opencv-python
  - numpy

Place the file `face_landmarker.task` in the project root.
Download it from:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# =============================================================================
# Project‑relative path to the face landmarker model file
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FACE_LANDMARKER_PATH = _PROJECT_ROOT / "face_landmarker.task"

if not _FACE_LANDMARKER_PATH.exists():
    raise FileNotFoundError(
        f"Face Landmarker task file not found: {_FACE_LANDMARKER_PATH}\n"
        "Please download it from https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )


class LipScorer:
    """
    Lip‑activity estimator using MediaPipe Face Landmarker.

    For each video frame, it extracts 478 face landmarks (including the
    inner lip contour), computes the Mouth Aspect Ratio (MAR) and lip
    motion, then fuses them into a single score in [0, 1] that correlates
    with speaking activity.
    """

    # Indices of inner lip landmarks (closed loop around the inner mouth)
    INNER_LIP_INDICES = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95
    ]

    def __init__(self, smooth_window: int = 3, motion_threshold: float = 0.001):
        """
        Args:
            smooth_window: number of frames for moving‑average smoothing.
            motion_threshold: minimum average landmark displacement (pixels)
                to consider the mouth as moving (helps suppress false positives).
        """
        self.smooth_window = smooth_window
        self.motion_threshold = motion_threshold

        # Initialise MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path=str(_FACE_LANDMARKER_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,                           # we track only one face
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _compute_mar(self, landmarks: dict) -> float:
        """
        Mouth Aspect Ratio from inner lip landmarks.

        MAR = average of three vertical distances / horizontal distance.
        A higher value indicates a more open mouth.

        Args:
            landmarks: dictionary mapping landmark index → (x, y, z) in pixels.

        Returns:
            MAR value (typically 0.0–0.8 for speech).
        """
        vert_pairs = [(13, 14), (82, 87), (312, 317)]
        horiz_pair = (78, 308)

        def distance(p1, p2):
            return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

        vert_dists = [distance(landmarks[a], landmarks[b]) for a, b in vert_pairs]
        horiz_dist = distance(landmarks[horiz_pair[0]], landmarks[horiz_pair[1]])

        if horiz_dist < 1e-6:
            return 0.0
        return (sum(vert_dists) / len(vert_dists)) / horiz_dist

    def process_video(self, video_path: str) -> np.ndarray:
        """
        Process a video file and return a 1D array of per‑frame lip‑activity scores.

        Args:
            video_path: path to the input video file.

        Returns:
            NumPy array of float scores in [0, 1], one element per frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        scores = []
        prev_landmarks = None
        frame_idx = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # MediaPipe requires RGB input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect landmarks (timestamp in ms, ~30 fps → 33 ms/frame)
            result = self.landmarker.detect_for_video(mp_image, frame_idx * 33)

            if result.face_landmarks:
                face_lm = result.face_landmarks[0]
                h, w, _ = frame.shape

                # Extract inner lip landmarks in pixel coordinates
                landmarks = {}
                for idx in self.INNER_LIP_INDICES:
                    lm = face_lm[idx]
                    landmarks[idx] = (lm.x * w, lm.y * h, lm.z)

                # 1. Mouth openness (MAR)
                mar = self._compute_mar(landmarks)

                # 2. Lip motion (average pixel displacement since previous frame)
                motion = 0.0
                if prev_landmarks is not None:
                    disp_sum = 0.0
                    count = 0
                    for idx in self.INNER_LIP_INDICES:
                        if idx in landmarks and idx in prev_landmarks:
                            p1 = np.array(landmarks[idx][:2])
                            p2 = np.array(prev_landmarks[idx][:2])
                            disp_sum += np.linalg.norm(p1 - p2)
                            count += 1
                    if count > 0:
                        motion = disp_sum / count

                # Normalise components to [0, 1]
                mar_norm = min(mar / 0.8, 1.0)          # typical max MAR ~0.8
                motion_norm = min(motion / 0.005, 1.0)  # 0.005 px/frame ≈ small motion

                # Fuse (60 % openness, 40 % motion)
                score = 0.6 * mar_norm + 0.4 * motion_norm
                score = max(0.0, min(score, 1.0))

                # Suppress non‑speech frames
                if mar < 0.05 and motion < self.motion_threshold:
                    score = 0.0

                prev_landmarks = landmarks.copy()
            else:
                # No face detected → score = 0
                score = 0.0
                prev_landmarks = None

            scores.append(score)

        cap.release()

        # Smooth scores with a moving average
        scores = np.array(scores)
        if len(scores) > 0 and self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            scores = np.convolve(scores, kernel, mode='same')
            scores = np.clip(scores, 0.0, 1.0)

        return scores