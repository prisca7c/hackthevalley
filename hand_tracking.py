import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import json

# from playsound import playsound
import threading

from match_chord import match_chord


# FOR THE CHORD DETECTION
import chord_detect 
def chord_detected():
    print("Chord event triggered!")
    chord_detect.stop_listening()

# Run audio in background thread
audio_thread = threading.Thread(
    target=chord_detect.start_listening, 
    kwargs={"device_index": 1, "on_chord_detected": chord_detected},
    daemon=True
)



# def play_success_sound():
#     threading.Thread(target=playsound, args=('success.wav',), daemon=True).start()

def play_success_sound():
    """Try to play a success sound if playsound is available, otherwise no-op."""
    try:
        from playsound import playsound
        threading.Thread(target=playsound, args=('success.wav',), daemon=True).start()
    except Exception:
        # playsound not installed or file missing; silently ignore
        return

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

AI_INTERVAL = 5
last_ai_call = time.time()
ai_feedback = ""

# Guitar fretboard parameters
NUM_STRINGS = 6
NUM_FRETS = 12
STRING_NAMES = ["E", "B", "G", "D", "A", "E"]  # High to low
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]

# Chord library for AI tutor
CHORD_LIBRARY = {
    "C": {"frets": [None, 3, 2, 0, 1, 0], "fingers": [None, 3, 2, None, 1, None], "name": "C Major"},
    "D": {"frets": [None, None, 0, 2, 3, 2], "fingers": [None, None, None, 1, 3, 2], "name": "D Major"},
    "E": {"frets": [0, 2, 2, 1, 0, 0], "fingers": [None, 2, 3, 1, None, None], "name": "E Major"},
    "G": {"frets": [3, 2, 0, 0, 0, 3], "fingers": [3, 2, None, None, None, 4], "name": "G Major"},
    "A": {"frets": [None, 0, 2, 2, 2, 0], "fingers": [None, None, 1, 2, 3, None], "name": "A Major"},
    "Em": {"frets": [0, 2, 2, 0, 0, 0], "fingers": [None, 2, 3, None, None, None], "name": "E Minor"},
    "Am": {"frets": [None, 0, 2, 2, 1, 0], "fingers": [None, None, 2, 3, 1, None], "name": "A Minor"},
    "Dm": {"frets": [None, None, 0, 2, 3, 1], "fingers": [None, None, None, 2, 4, 1], "name": "D Minor"},
}

@dataclass
class FretboardRegion:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    fret_markers: List[Tuple[int, int, int, int]]
    string_positions: Optional[List[int]] = None
    confidence: float = 1.0
    quad_corners: Optional[np.ndarray] = None

@dataclass
class FingerPosition:
    finger_name: str
    string_num: int
    fret_num: int
    timestamp: float

class StabilizedTracker:
    def __init__(self, smoothing_factor=0.85, history_size=8):
        self.smoothing_factor = smoothing_factor
        self.history = deque(maxlen=history_size)
        self.current_value = None
    def update(self, new_value):
        if new_value is None:
            return self.current_value
        self.history.append(new_value)
        if self.current_value is None:
            self.current_value = new_value
        else:
            self.current_value = (self.smoothing_factor * self.current_value + 
                                 (1 - self.smoothing_factor) * new_value)
        return self.current_value
    def reset(self):
        self.history.clear()
        self.current_value = None

class FretboardDetector:
    def __init__(self):
        print("[FretboardDetector] __init__")
        self.tracking_enabled = False
        self.feature_detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2)
        self.reference_features = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # make corner trackers smoother (higher smoothing -> less jitter)
        self.corner_smoothing = 0.1
        self.corner_trackers = [StabilizedTracker(smoothing_factor=self.corner_smoothing) for _ in range(4)]
        self.fret_trackers = []
        self.lost_track_frames = 0
        self.max_lost_frames = 8
        self.last_valid_region = None
        # stability checks to avoid jitter: tuned for responsiveness
        self.stability_count = 0
        # require a couple of consecutive small-motion frames before accepting non-LK homographies
        self.required_stable_frames = 2
        # tighter per-frame motion threshold (px) to avoid jitter
        self.stability_threshold_px = 16
        # minimum homography inliers required (raise to be more robust)
        self.min_inliers = 8
        # lightweight debug timer to log match/inlier counts once per second
        self._last_debug_time = 0
        self.debug = False
        # Lucas-Kanade optical flow fallback
        self.use_lk = False
        self.lk_prev_gray = None
        self.lk_points = None
        self.lk_src_pts = None
        # LK params tuned for reasonably stable tracking
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    def set_reference_region(self, frame, region: FretboardRegion):
        print("[FretboardDetector] set_reference_region called")
# src_pts: your 4 detected fretboard corners
        (x1, y1) = region.corners[0]
        (x2, y2) = region.corners[1]
        (x3, y3) = region.corners[2]
        (x4, y4) = region.corners[3]

        src_pts = np.float32([
            [x1, y1], [x2, y2], [x3, y3], [x4, y4]
        ])

        # Choose desired rectified output size
        width, height = 800, 200
        dst_pts = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        # Perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp trapezoid -> rectangle
        reference_patch = cv2.warpPerspective(frame, M, (width, height))
        if reference_patch.size == 0:
            return
        gray_patch = cv2.cvtColor(reference_patch, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_patch)
        self.reference_keypoints, self.reference_descriptors = self.feature_detector.detectAndCompute(enhanced, None)
        # handle case when ORB finds no keypoints (detectAndCompute may return None)
        if self.reference_keypoints is None or len(self.reference_keypoints) == 0:
            # fallback: don't fail — keep descriptors None and allow LK-only tracking
            self.reference_keypoints = []
            self.reference_descriptors = None

        self.reference_features = {
            'patch': reference_patch.copy(),
            'region': region,
            'width': width,        # warped fretboard width
            'height': height,      # warped fretboard height
            'src_pts': src_pts,    # original 4 corner points
            'dst_pts': dst_pts,    # rectified rectangle corners
            'M': M,                # forward perspective transform
            'M_inv': cv2.getPerspectiveTransform(dst_pts, src_pts),  # inverse transform
            # offset for any relative coordinate calculations (x, y)
            'offset': (region.top_left[0], region.top_left[1])
        }
        
        self.tracking_enabled = True
        # initialize corner trackers with the detected quadrilateral corners (for display smoothing)
        try:
            quad = np.float32([region.corners[0], region.corners[1], region.corners[2], region.corners[3]])
        except Exception:
            quad = np.float32([
                [region.top_left[0], region.top_left[1]],
                [region.bottom_right[0], region.top_left[1]],
                [region.bottom_right[0], region.bottom_right[1]],
                [region.top_left[0], region.bottom_right[1]]
            ])
        for i, tracker in enumerate(self.corner_trackers):
            tracker.update(quad[i])
        self.fret_trackers = []
        if region.fret_markers:
            for marker in region.fret_markers:
                self.fret_trackers.append(StabilizedTracker(smoothing_factor=0.92))
        self.last_valid_region = region
        # Dense LK initialization: detect good features to track inside the rectified reference patch
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.lk_prev_gray = gray
            # detect good features on the rectified reference_patch (dst coordinate space)
            ref_gray = gray_patch  # already the rectified gray patch
            max_corners = 250
            corners = cv2.goodFeaturesToTrack(ref_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=6)
            if corners is not None and len(corners) >= 4:
                # corners are in rectified patch coordinates (x,y). Transform to image coords using inverse map
                # corners shape Nx1x2 float32
                self.lk_src_pts = corners.astype(np.float32)
                # transform to image coordinates using stored inverse transform
                try:
                    M_inv = self.reference_features.get('M_inv')
                    if M_inv is not None:
                        img_pts = cv2.perspectiveTransform(self.lk_src_pts, M_inv)
                        self.lk_points = img_pts.astype(np.float32)
                        self.use_lk = True
                    else:
                        self.lk_points = None
                        self.use_lk = False
                except Exception:
                    self.lk_points = None
                    self.use_lk = False
            else:
                self.lk_src_pts = None
                self.lk_points = None
                self.use_lk = False
        except Exception:
            self.lk_src_pts = None
            self.lk_points = None
            self.use_lk = False
        kp_count = len(self.reference_keypoints) if self.reference_keypoints else 0
        if kp_count == 0:
            print("✓ Tracking initialized (0 ORB features) — LK mode: {}".format(self.use_lk))
        else:
            print(f"✓ Tracking initialized with {kp_count} features — LK mode: {self.use_lk}")
    def track_region(self, frame, last_region: FretboardRegion) -> Optional[FretboardRegion]:
        print("[FretboardDetector] track_region called (tracking_enabled=%s, has_reference=%s)" % (
            str(self.tracking_enabled), str(self.reference_features is not None)
        ))
        # allow LK-only tracking: require reference_features initialized (which contains M and LK points)
        if not self.tracking_enabled or self.reference_features is None:
            print("[FretboardDetector] track_region: tracking disabled or no reference features")
            return last_region
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_frame = clahe.apply(gray_frame)
        current_keypoints, current_descriptors = self.feature_detector.detectAndCompute(enhanced_frame, None)
        low_features = (current_descriptors is None or len(current_keypoints) < 15)

        # do not return immediately on low features: try Lucas-Kanade before failing
        self.lost_track_frames = 0
    # Try Lucas-Kanade optical flow first if initialized
        if self.use_lk and self.lk_prev_gray is not None and self.lk_points is not None:
            try:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.lk_prev_gray, gray_frame, self.lk_points, None, **self.lk_params)
                # require that most points are tracked successfully
                if next_pts is not None and status is not None and status.sum() >= max(2, int(0.5 * len(self.lk_points))):
                    tracked_pts = next_pts.reshape(-1, 2)
                    dst_pts_rect = self.reference_features['dst_pts']
                    # compute transform from rectified patch -> current frame using tracked points
                    try:
                        M_lk = cv2.getPerspectiveTransform(dst_pts_rect, tracked_pts.astype(np.float32))
                        M = M_lk
                        lk_used = True
                        # update LK buffers
                        self.lk_points = next_pts
                        self.lk_prev_gray = gray_frame
                        # continue with the rest of the pipeline using M
                    except Exception:
                        M = None
                else:
                    # LK insufficient - fall back to feature matching
                    M = None
            except Exception:
                M = None
        else:
            M = None
        # If LK didn't produce M, fall back to descriptor matching and compute homography from matches
        if M is None:
            # if descriptors are missing and LK failed, we can't proceed
            if low_features:
                self.lost_track_frames += 1
                if self.lost_track_frames > self.max_lost_frames:
                    print("⚠ Tracking lost - not enough features and LK failed")
                return last_region
            matches = self.matcher.knnMatch(self.reference_descriptors, current_descriptors, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
            # If a hand enters the frame, many matches can come from skin regions.
            # Filter out matches whose current image locations fall on skin-colored pixels
            try:
                # only apply skin filtering when a significant fraction of matches fall on skin
                ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                skin_filtered = []
                skin_count = 0
                for m in good_matches:
                    kp = current_keypoints[m.trainIdx].pt
                    xk, yk = int(kp[0]), int(kp[1])
                    if xk < 0 or yk < 0 or yk >= ycrcb_img.shape[0] or xk >= ycrcb_img.shape[1]:
                        continue
                    cr = int(ycrcb_img[yk, xk, 1])
                    cb = int(ycrcb_img[yk, xk, 2])
                    # Typical skin thresholds in YCrCb
                    if 133 <= cr <= 173 and 77 <= cb <= 127:
                        # likely skin
                        skin_count += 1
                        continue
                    skin_filtered.append(m)
                if len(good_matches) > 0:
                    frac_skin = skin_count / float(len(good_matches))
                else:
                    frac_skin = 0.0
                # if a large fraction of matches are on skin, use the filtered set; otherwise keep original
                if frac_skin > 0.35 and len(skin_filtered) >= self.min_inliers:
                    good_matches = skin_filtered
            except Exception:
                # if any error, keep original good_matches
                pass
            if len(good_matches) < self.min_inliers:
                self.lost_track_frames += 1
                return last_region
            src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches])
            # compute homography from matched points
            if len(src_pts) >= 4 and len(dst_pts) >= 4:
                M, mask = cv2.findHomography(src_pts.reshape(-1,1,2), dst_pts.reshape(-1,1,2), cv2.RANSAC, 5.0)
            else:
                M = None
            if M is None:
                return last_region
            # require a minimum number of inliers to trust the homography
            inliers = 0
            try:
                inliers = int(mask.sum()) if mask is not None else 0
            except Exception:
                try:
                    inliers = int(np.sum(mask))
                except Exception:
                    inliers = 0
            if inliers < self.min_inliers:
                self.lost_track_frames += 1
                return last_region
            lk_used = False
        else:
            lk_used = True
        ref_width = self.reference_features['width']
        ref_height = self.reference_features['height']
        offset_x, offset_y = self.reference_features['offset']
        corners = np.float32([
            [0, 0],
            [ref_width, 0],
            [ref_width, ref_height],
            [0, ref_height]
        ]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        # sanity-check transformed corners against last valid region to avoid insane jumps
        if self.last_valid_region is not None and hasattr(self.last_valid_region, 'quad_corners') and self.last_valid_region.quad_corners is not None:
            prev = self.last_valid_region.quad_corners.reshape(-1, 2)
            cur = transformed_corners.reshape(-1, 2)
            # compute euclidean distances
            dists = np.linalg.norm(cur - prev, axis=1)
            # if any corner moves more than a very large threshold suddenly, reject this frame
            large_jump_thresh = max(120, 0.25 * max(frame.shape[0], frame.shape[1]))
            if np.any(dists > large_jump_thresh):
                # unstable mapping - ignore
                self.lost_track_frames += 1
                return last_region
            # If transform came from LK, accept it immediately for fluency
            if lk_used:
                stable_enough = True
            else:
                # check for small motion: accumulate stability count only if median corner motion is small
                median_dist = float(np.median(dists))
                if median_dist < self.stability_threshold_px:
                    self.stability_count += 1
                else:
                    self.stability_count = 0
                # require several consecutive small-motion frames before accepting
                stable_enough = (self.stability_count >= self.required_stable_frames)
                if not stable_enough:
                    # accept quicker when homography is supported by many inliers and median motion is modest
                    try:
                        many_inliers = (inliers >= max(20, self.min_inliers * 3))
                    except Exception:
                        many_inliers = False
                    if many_inliers and median_dist < max(self.stability_threshold_px * 2, 16):
                        stable_enough = True
            # optional debug overlay: draw previous and current corner positions
        else:
            # no previous region: treat as stable to initialize
            stable_enough = True
            dists = np.zeros((4,))
        if self.debug:
            try:
                # draw previous quad (blue) and new transformed corners (red)
                prev = self.last_valid_region.quad_corners.astype(np.int32)
                for p in prev:
                    cv2.circle(frame, tuple(p[0]), 3, (255, 0, 0), -1)
                cur = transformed_corners.astype(np.int32)
                for p in cur:
                    cv2.circle(frame, tuple(p[0]), 4, (0, 0, 255), -1)
            except Exception:
                pass
        else:
            # no previous region: treat as stable to initialize
            stable_enough = True
            dists = np.zeros((4,))
        smoothed_corners = []
        for i, corner in enumerate(transformed_corners):
            smoothed = self.corner_trackers[i].update(corner[0])
            if smoothed is not None:
                smoothed_corners.append(smoothed)
            else:
                smoothed_corners.append(corner[0])
        smoothed_corners = np.array(smoothed_corners).reshape(-1, 1, 2)
        h, w = frame.shape[:2]
        all_in_bounds = all(
            0 <= pt[0, 0] <= w and 0 <= pt[0, 1] <= h 
            for pt in smoothed_corners
        )
        if not all_in_bounds:
            return last_region
        new_markers = []
        if last_region.fret_markers:
            for marker in last_region.fret_markers:
                rel_x = (marker[0] - self.reference_features['region'].top_left[0]) / (
                    self.reference_features['region'].bottom_right[0] - 
                    self.reference_features['region'].top_left[0]
                )
                fret_points = np.float32([
                    [rel_x * ref_width, 0],
                    [rel_x * ref_width, ref_height]
                ]).reshape(-1, 1, 2)
                transformed_fret = cv2.perspectiveTransform(fret_points, M)
                new_x = int(transformed_fret[0, 0, 0])
                new_y_top = int(transformed_fret[0, 0, 1])
                new_y_bottom = int(transformed_fret[1, 0, 1])
                new_markers.append((new_x, new_y_top, 2, new_y_bottom - new_y_top))
        # If the mapping isn't stable yet, keep returning the last_region (avoids jitter)
        if not stable_enough:
            return last_region

        tracked_region = FretboardRegion(
            top_left=(int(smoothed_corners[0, 0, 0]), int(smoothed_corners[0, 0, 1])),
            bottom_right=(int(smoothed_corners[2, 0, 0]), int(smoothed_corners[2, 0, 1])),
            fret_markers=new_markers,
            confidence=len(good_matches) / max(len(self.reference_keypoints), 1)
        )
        tracked_region.quad_corners = smoothed_corners
        # update last valid region for future stability checks
        self.last_valid_region = tracked_region
        # reset stability count so subsequent small motions are re-evaluated
        self.stability_count = 0
        return tracked_region

class FingerTracker:
    def __init__(self, log_interval=0.3):
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.current_positions: List[FingerPosition] = []
    def detect_position(self, tip_x, tip_y, region: FretboardRegion) -> Optional[Tuple[int, int]]:
        if hasattr(region, 'quad_corners') and region.quad_corners is not None:
            corners = region.quad_corners
            result = cv2.pointPolygonTest(corners.reshape(-1, 2).astype(np.int32), (tip_x, tip_y), False)
            if result < 0:
                return None
            left_top = corners[0, 0]
            left_bottom = corners[3, 0]
            right_top = corners[1, 0]
            right_bottom = corners[2, 0]
            def point_to_line_t(p, line_start, line_end):
                line_vec = line_end - line_start
                point_vec = p - line_start
                line_len_sq = np.dot(line_vec, line_vec)
                if line_len_sq == 0:
                    return 0
                t = np.dot(point_vec, line_vec) / line_len_sq
                return max(0, min(1, t))
            point_arr = np.array([tip_x, tip_y])
            t_left = point_to_line_t(point_arr, left_top, left_bottom)
            t_right = point_to_line_t(point_arr, right_top, right_bottom)
            t_string = (t_left + t_right) / 2
            string_idx = int(t_string * NUM_STRINGS)
            string_idx = max(0, min(NUM_STRINGS - 1, string_idx))
            if region.fret_markers:
                min_x = min(corners[:, 0, 0])
                fret_x_positions = [min_x] + [m[0] for m in region.fret_markers]
                for i in range(len(fret_x_positions) - 1):
                    if fret_x_positions[i] <= tip_x <= fret_x_positions[i + 1]:
                        return string_idx + 1, i + 1
                if tip_x > fret_x_positions[-1]:
                    return string_idx + 1, len(fret_x_positions)
            return None
        tl_x, tl_y = region.top_left
        br_x, br_y = region.bottom_right
        if not (tl_x <= tip_x <= br_x and tl_y <= tip_y <= br_y):
            return None
        string_height = (br_y - tl_y) / NUM_STRINGS
        string_idx = int((tip_y - tl_y) / string_height)
        string_idx = max(0, min(NUM_STRINGS - 1, string_idx))
        if region.fret_markers:
            fret_x_positions = [tl_x - 10] + [m[0] for m in region.fret_markers]
            for i in range(len(fret_x_positions) - 1):
                if fret_x_positions[i] <= tip_x <= fret_x_positions[i + 1]:
                    return string_idx + 1, i + 1
            if tip_x > fret_x_positions[-1]:
                return string_idx + 1, len(fret_x_positions)
        return None
    def update_positions(self, hand_landmarks, image_width, image_height, region: Optional[FretboardRegion]) -> List[Tuple[int, int, FingerPosition]]:
        if not region:
            return []
        current_time = time.time()
        should_log = current_time - self.last_log_time >= self.log_interval
        if should_log:
            self.current_positions.clear()
            self.last_log_time = current_time
        positions = []
        finger_tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_mcps = [
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.PINKY_MCP
        ]
        for i, (tip_idx, mcp_idx) in enumerate(zip(finger_tips, finger_mcps)):
            tip = hand_landmarks.landmark[tip_idx]
            mcp = hand_landmarks.landmark[mcp_idx]
            tip_x = int(tip.x * image_width)
            tip_y = int(tip.y * image_height)
            if tip.z < mcp.z - 0.02:
                result = self.detect_position(tip_x, tip_y, region)
                if result:
                    string_num, fret_num = result
                    position = FingerPosition(
                        finger_name=FINGER_NAMES[i],
                        string_num=string_num,
                        fret_num=fret_num,
                        timestamp=current_time
                    )
                    if should_log:
                        self.current_positions.append(position)
                    positions.append((tip_x, tip_y, position))
        return positions

def draw_hand_guide(frame, x, y, hand='L'):
    cv2.ellipse(frame, (x+40, y+60), (30, 35), 0, 0, 360, (200,200,200), -1)
    for i in range(4):
        fx = x + 20 + i*15
        cv2.line(frame, (fx, y+30), (fx, y+60), (180,180,180), 8)
        cv2.putText(frame, str(i+1), (fx-5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.line(frame, (x+15, y+70), (x+5, y+90), (180,180,180), 8)
    cv2.putText(frame, hand, (x+60, y+100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

class ChordTutor:
    def __init__(self):
        self.current_chord = None
        self.chord_history = deque(maxlen=10)
    def set_chord(self, chord_name: str):
        if chord_name in CHORD_LIBRARY:
            self.current_chord = chord_name
            return True
        return False
    def check_chord_accuracy(self, finger_positions: List[FingerPosition]) -> Dict:
        if not self.current_chord:
            return {"status": "no_chord_selected"}
        chord_info = CHORD_LIBRARY[self.current_chord]
        required_positions = []
        for string_idx, fret in enumerate(chord_info['frets']):
            if fret is not None:
                required_positions.append((string_idx + 1, fret))
        user_positions = [(p.string_num, p.fret_num) for p in finger_positions]
        correct = []
        missing = []
        wrong = []
        for req in required_positions:
            if req in user_positions:
                correct.append(req)
            else:
                missing.append(req)
        for pos in user_positions:
            if pos not in required_positions:
                wrong.append(pos)
        accuracy = len(correct) / len(required_positions) if required_positions else 0
        return {
            "status": "checking",
            "accuracy": accuracy,
            "correct": correct,
            "missing": missing,
            "wrong": wrong,
            "is_perfect": len(missing) == 0 and len(wrong) == 0
        }
    def draw_chord_diagram(self, frame, x, y, width, height, show_instructions=True):
        if not self.current_chord:
            return
        chord_info = CHORD_LIBRARY[self.current_chord]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, chord_info['name'], (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        diagram_x = x + 20
        diagram_y = y + 45
        diagram_w = width - 40
        diagram_h = height - 65
        string_spacing = diagram_w // (NUM_STRINGS - 1)
        for i in range(NUM_STRINGS):
            sx = diagram_x + i * string_spacing
            cv2.line(frame, (sx, diagram_y), (sx, diagram_y + diagram_h), (200, 200, 200), 1)
        fret_spacing = diagram_h // 5
        for i in range(6):
            fy = diagram_y + i * fret_spacing
            thickness = 3 if i == 0 else 1
            cv2.line(frame, (diagram_x, fy), (diagram_x + diagram_w, fy), (200, 200, 200), thickness)
        for string_idx, fret in enumerate(chord_info['frets']):
            sx = diagram_x + string_idx * string_spacing
            if fret is None:
                cv2.putText(frame, "X", (sx - 5, diagram_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
            elif fret == 0:
                cv2.circle(frame, (sx, diagram_y - 10), 6, (100, 255, 100), 2)
            else:
                fy = diagram_y + (fret - 0.5) * fret_spacing
                cv2.circle(frame, (int(sx), int(fy)), 6, (0, 255, 255), -1)
                cv2.circle(frame, (int(sx), int(fy)), 6, (255, 255, 255), 2)
                finger_num = chord_info['fingers'][string_idx]
                if finger_num:
                    cv2.putText(frame, str(finger_num), (int(sx) - 5, int(fy) + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        if show_instructions:
            instructions = []
            for string_idx, fret in enumerate(chord_info['frets']):
                finger = chord_info['fingers'][string_idx]
                if fret is not None and fret > 0 and finger:
                    instructions.append(f"Finger {finger} on string {string_idx+1} fret {fret}")
            if not instructions:
                instructions = ["Open strings or muted"]
            for i, line in enumerate(instructions):
                cv2.putText(frame, line, (x + 10, y + height - 35 + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def draw_fretboard(image, region: FretboardRegion, debug=False, chord=None, show_grid_fingers=True):
    if hasattr(region, 'quad_corners') and region.quad_corners is not None:
        corners = region.quad_corners.astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        for i in range(NUM_STRINGS):
            t = (i + 0.5) / NUM_STRINGS
            left_point = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
            right_point = corners[1, 0] + t * (corners[2, 0] - corners[1, 0])
            cv2.line(image, 
                    (int(left_point[0]), int(left_point[1])),
                    (int(right_point[0]), int(right_point[1])),
                    (0, 200, 0), 1)
            label_x = int(left_point[0] - 25)
            label_y = int(left_point[1] + 5)
            cv2.putText(image, STRING_NAMES[i], (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for fret in range(1, NUM_FRETS):
            t = fret / NUM_FRETS
            top = corners[0, 0] + t * (corners[1, 0] - corners[0, 0])
            bottom = corners[3, 0] + t * (corners[2, 0] - corners[3, 0])
            cv2.line(image, 
                     (int(top[0]), int(top[1])),
                     (int(bottom[0]), int(bottom[1])),
                     (0, 200, 0), 2)
            cv2.putText(image, str(fret), (int(top[0]) - 8, int(top[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(image, 
                 (int(corners[0, 0][0]), int(corners[0, 0][1])),
                 (int(corners[3, 0][0]), int(corners[3, 0][1])),
                 (0, 255, 0), 4)
        if chord is not None:
            chord_info = CHORD_LIBRARY[chord]
            for string_idx, (fret, finger_num) in enumerate(zip(chord_info['frets'], chord_info['fingers'])):
                if fret is not None and fret > 0 and finger_num:
                    t = (string_idx + 0.5) / NUM_STRINGS
                    string_start = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
                    string_end = corners[1, 0] + t * (corners[2, 0] - corners[1, 0])
                    fret_t = fret / NUM_FRETS
                    sx = int(string_start[0] + (string_end[0] - string_start[0]) * fret_t)
                    sy = int(string_start[1] + (string_end[1] - string_start[1]) * fret_t)
                    cv2.circle(image, (sx, sy), 8, (0, 255, 255), -1)
                    cv2.circle(image, (sx, sy), 8, (255, 255, 255), 2)
                    if show_grid_fingers:
                        cv2.putText(image, str(finger_num), (sx - 7, sy + 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                elif fret == 0:
                    t = (string_idx + 0.5) / NUM_STRINGS
                    string_start = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
                    sx = int(string_start[0])
                    sy = int(string_start[1])
                    cv2.circle(image, (sx, sy - 18), 8, (100, 255, 100), 2)
                elif fret is None:
                    t = (string_idx + 0.5) / NUM_STRINGS
                    string_start = corners[0, 0] + t * (corners[3, 0] - corners[0, 0])
                    sx = int(string_start[0])
                    sy = int(string_start[1])
                    cv2.putText(image, "X", (sx - 8, sy - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        if debug:
            for corner in corners:
                cv2.circle(image, tuple(corner[0]), 5, (255, 0, 0), -1)
    else:
        tl_x, tl_y = region.top_left
        br_x, br_y = region.bottom_right
        cv2.rectangle(image, region.top_left, region.bottom_right, (0, 255, 0), 2)
        string_height = (br_y - tl_y) / NUM_STRINGS
        for i in range(NUM_STRINGS + 1):
            y = int(tl_y + i * string_height)
            cv2.line(image, (tl_x, y), (br_x, y), (0, 200, 0), 1)
            if i < NUM_STRINGS:
                mid_y = int(tl_y + (i + 0.5) * string_height)
                cv2.putText(image, STRING_NAMES[i], (tl_x - 25, mid_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if region.fret_markers:
            for i, (x, y, w, h) in enumerate(region.fret_markers):
                cv2.line(image, (x, tl_y), (x, br_y), (0, 200, 0), 2)
                cv2.putText(image, str(i + 1), (x - 8, tl_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        pass

def save_positions(positions: List[FingerPosition], filename="guitar_positions.txt"):
    with open(filename, "w") as f:
        f.write(f"Guitar Finger Positions - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 60 + "\n\n")
        if not positions:
            f.write("No fingers detected on fretboard\n")
        else:
            for pos in positions:
                note = f"{STRING_NAMES[pos.string_num - 1]}"
                f.write(f"{pos.finger_name:8} → String {pos.string_num} ({note}), "
                       f"Fret {pos.fret_num}\n")

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = FretboardDetector()
    tracker = FingerTracker()
    tutor = ChordTutor()
    manual_mode = True
    manual_region: Optional[FretboardRegion] = None
    temp_coords = []
    current_frame = None
    debug_mode = False
    show_tracking_info = True
    show_tutor_panel = True
    show_hand_guide = True
    show_grid_fingers = True
    tutor.set_chord("C")
    available_chords = list(CHORD_LIBRARY.keys())
    current_chord_idx = 0

    chord_completed = False
    checkmark_time = 0

    print("=" * 60)
    print("🎸 AI Guitar Tutor with Advanced Tracking")
    print("=" * 60)
    print("\nSetup:")
    print("  Click the FOUR corners of the fretboard in order: top-left, top-right, bottom-right, bottom-left")
    print("  (Right-click to undo the last point. Press 'r' to reset selection.)")
    print("  3. Practice the displayed chord!")
    print("\nControls:")
    print("  'd' - Toggle debug mode")
    print("  't' - Toggle tracking info")
    print("  'u' - Toggle tutor panel")
    print("  'h' - Toggle hand guide")
    print("  'f' - Toggle grid finger numbers")
    print("  'n' - Next chord")
    print("  'p' - Previous chord")
    print("  'r' - Reset fretboard selection")
    print("  SPACE - Next chord after checkmark")
    print("  ESC - Exit")
    print("=" * 60 + "\n")

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]     # top-left
        rect[2] = pts[np.argmax(s)]     # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def mouse_callback(event, x, y, flags, param):
        nonlocal temp_coords, manual_region, manual_mode, current_frame
        # debug: report clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"[mouse_callback] LMB at {(x,y)} manual_mode={manual_mode} temp_len={len(temp_coords)}")
        # If not in manual mode, allow a click to enter manual selection and start the first corner
        if event == cv2.EVENT_LBUTTONDOWN and not manual_mode:
            manual_mode = True
            temp_coords = [(x, y)]
            print(f"[mouse_callback] Entered manual selection; first corner {(x,y)}")
            return
        # Left-click: add a corner (when already in manual mode)
        if event == cv2.EVENT_LBUTTONDOWN and manual_mode:
            if len(temp_coords) < 4:
                temp_coords.append((x, y))
            else:
                # ignore extra clicks until reset
                return
            if len(temp_coords) == 4:
                # interpret the four user clicks as the quad corners
                pts = np.float32([temp_coords[0], temp_coords[1], temp_coords[2], temp_coords[3]])
                pts_ordered = order_points(pts)

                x_vals = [int(p[0]) for p in pts_ordered]
                y_vals = [int(p[1]) for p in pts_ordered]
                tl = (min(x_vals), min(y_vals))
                br = (max(x_vals), max(y_vals))
                fret_width = (br[0] - tl[0]) / NUM_FRETS
                markers = [(int(tl[0] + i * fret_width), tl[1], 2, br[1] - tl[1])
                          for i in range(1, NUM_FRETS + 1)]
                manual_region = FretboardRegion(
                    top_left=tl,
                    bottom_right=br,
                    fret_markers=markers
                )
                # store the ordered corner points on the region so set_reference_region can use them
                try:
                    manual_region.corners = pts_ordered.tolist()
                except Exception:
                    manual_region.corners = [tuple(pt) for pt in pts_ordered]
                # also store quad_corners in same format used elsewhere
                manual_region.quad_corners = pts_ordered.reshape(-1, 1, 2)

                if current_frame is not None:
                    print("[mouse_callback] calling set_reference_region immediately")
                    detector.set_reference_region(current_frame, manual_region)
                else:
                    # if user clicked before first frame available, queue it and main loop will apply
                    print("[mouse_callback] queuing manual_region until frame available")
                    queued_manual_region.append(manual_region)
                manual_mode = False
                temp_coords.clear()
                print("✓ Fretboard tracking enabled")
        # Right-click: undo last point
        elif event == cv2.EVENT_RBUTTONDOWN and manual_mode:
            if temp_coords:
                temp_coords.pop()
    cv2.namedWindow('AI Guitar Tutor')
    cv2.setMouseCallback('AI Guitar Tutor', mouse_callback)
    # queue in case mouse callback happens before we have a frame
    queued_manual_region: List[FretboardRegion] = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        current_frame = frame.copy()
        # if a manual region was queued (clicked before first frame), initialize detector now
        if queued_manual_region:
            try:
                queued = queued_manual_region.pop(0)
                print("[main] initializing detector from queued manual region")
                detector.set_reference_region(current_frame, queued)
            except Exception as e:
                print("[main] queued set_reference_region failed:", e)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # --- New Feature: Chord Confirmation ---
        if tracker.current_positions:
            # Extract finger positions
            finger_positions = [(FINGER_NAMES.index(pos.finger_name) + 1, pos.string_num, pos.fret_num) for pos in tracker.current_positions]
            print (finger_positions)
            # Match the chord
            matched_chord = match_chord(finger_positions)

            # Provide feedback
            if matched_chord:
                cv2.putText(frame, f"Chord: {matched_chord}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No matching chord", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


        # --- Draw MediaPipe hand skeleton and purple fingertips ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                tracker.update_positions(hand_landmarks, w, h, manual_region)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )
                fingertip_ids = [
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP
                ]
                for tip_id in fingertip_ids:
                    tip = hand_landmarks.landmark[tip_id]
                    cx, cy = int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
        # ---------------------------------------------------------

        active_region = manual_region
        if not manual_mode and manual_region is not None:
            detector.debug = debug_mode
            tracked_region = detector.track_region(frame, manual_region)
            if tracked_region:
                manual_region = tracked_region
                active_region = tracked_region
        if manual_mode:
            cv2.putText(frame, "Click FOUR corners (TL, TR, BR, BL). Right-click to undo.",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for i, pt in enumerate(temp_coords):
                cv2.circle(frame, pt, 6, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if active_region:
            # draw small status text about detector
            ref_status = 'YES' if detector.reference_features is not None else 'NO'
            det_status = 'ON' if detector.tracking_enabled else 'OFF'
            cv2.putText(frame, f'Detector: {det_status} Ref: {ref_status}', (10, h-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            draw_fretboard(frame, active_region, debug_mode, chord=tutor.current_chord, show_grid_fingers=show_grid_fingers)
            if show_tracking_info and active_region.confidence > 0:
                conf_text = f"Track: {active_region.confidence:.1%}"
                conf_color = (0, 255, 0) if active_region.confidence > 0.3 else (0, 165, 255)
                cv2.putText(frame, conf_text, (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                if detector.tracking_enabled and detector.lost_track_frames <= detector.max_lost_frames:
                    cv2.circle(frame, (w - 30, 30), 8, (0, 255, 0), -1)
                else:
                    cv2.circle(frame, (w - 30, 30), 8, (0, 0, 255), -1)
        info_panel_y = h - 200
        for i, pos in enumerate(tracker.current_positions[:4]):
            text = f"{pos.finger_name}: S{pos.string_num}({STRING_NAMES[pos.string_num-1]}) F{pos.fret_num}"
            cv2.putText(frame, text, (30, info_panel_y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if show_tutor_panel and not manual_mode:
            panel_width = 280
            panel_height = 300
            panel_x = w - panel_width - 10
            panel_y = 60
            tutor.draw_chord_diagram(frame, panel_x, panel_y, panel_width, panel_height, show_instructions=False)
            if show_hand_guide:
                draw_hand_guide(frame, panel_x + 170, panel_y + 170, hand='L')
            if tracker.current_positions:
                accuracy_result = tutor.check_chord_accuracy(tracker.current_positions)
                acc_y = panel_y + panel_height + 20
                bar_width = panel_width - 20
                bar_height = 25
                bar_x = panel_x + 10
                cv2.rectangle(frame, (bar_x, acc_y), (bar_x + bar_width, acc_y + bar_height),
                             (40, 40, 40), -1)
                accuracy_pct = accuracy_result['accuracy']
                filled_width = int(bar_width * accuracy_pct)
                color = (0, 255, 0) if accuracy_pct >= 0.8 else (0, 165, 255) if accuracy_pct >= 0.5 else (0, 100, 255)
                cv2.rectangle(frame, (bar_x, acc_y), (bar_x + filled_width, acc_y + bar_height),
                             color, -1)
                cv2.rectangle(frame, (bar_x, acc_y), (bar_x + bar_width, acc_y + bar_height),
                             (200, 200, 200), 2)
                acc_text = f"Accuracy: {int(accuracy_pct * 100)}%"
                cv2.putText(frame, acc_text, (bar_x, acc_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                feedback_y = acc_y + bar_height + 25

                # --- Chord completion logic ---
                if accuracy_result['is_perfect'] and not chord_completed:
                    audio_thread.start()
                    play_success_sound()
                    chord_completed = True
                    checkmark_time = time.time()

                if chord_completed:
                    # Draw big green checkmark and prompt
                    cv2.putText(frame, "✓", (bar_x + bar_width//2 - 20, feedback_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 8)
                    cv2.putText(frame, "Press SPACE for next chord", (bar_x, feedback_y + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif accuracy_result['wrong']:
                    cv2.putText(frame, "Check finger positions", (bar_x, feedback_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
            if tutor.current_chord:
                chord_info = CHORD_LIBRARY[tutor.current_chord]
                instructions = []
                for string_idx, fret in enumerate(chord_info['frets']):
                    finger = chord_info['fingers'][string_idx]
                    if fret is not None and fret > 0 and finger:
                        instructions.append(f"Finger {finger} on string {string_idx+1} fret {fret}")
                if not instructions:
                    instructions = ["Open strings or muted"]
                for i, line in enumerate(instructions):
                    cv2.putText(frame, line, (panel_x, panel_y + panel_height + 80 + i*28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if tracker.current_positions:
                    accuracy_result = tutor.check_chord_accuracy(tracker.current_positions)
                    # --- Green/Red light indicator ---
                    if accuracy_result['is_perfect']:
                        light_color = (0, 255, 0)  # Green
                    else:
                        light_color = (0, 0, 255)  # Red
                    cv2.circle(frame, (40, 40), 20, light_color, -1)
                    cv2.putText(frame, "FINGER", (70, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
        status_y = h - 70
        tracking_status = "Tracking: Active" if (not manual_mode and detector.tracking_enabled) else "Mode: Setup"
        tracking_color = (0, 255, 0) if detector.tracking_enabled else (0, 0, 255)
        cv2.rectangle(frame, (5, status_y - 35), (w - 5, h - 5), (30, 30, 30), -1)
        cv2.rectangle(frame, (5, status_y - 35), (w - 5, h - 5), (100, 100, 100), 2)
        cv2.putText(frame, tracking_status, (15, status_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)
        cv2.putText(frame, "N:Next P:Prev R:Reset U:UI D:Debug T:Track H:Hand F:Fingers SPACE:Next ESC:Exit",
                   (15, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow('AI Guitar Tutor', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            manual_mode = True
            manual_region = None
            temp_coords.clear()
            detector.tracking_enabled = False
            detector.reference_features = None
            for t in detector.corner_trackers:
                t.reset()
            chord_completed = False
            print("Reset - select new fretboard area")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('t'):
            show_tracking_info = not show_tracking_info
            print(f"Tracking info: {'ON' if show_tracking_info else 'OFF'}")
        elif key == ord('u'):
            show_tutor_panel = not show_tutor_panel
            print(f"Tutor panel: {'ON' if show_tutor_panel else 'OFF'}")
        elif key == ord('h'):
            show_hand_guide = not show_hand_guide
            print(f"Hand guide: {'ON' if show_hand_guide else 'OFF'}")
        elif key == ord('f'):
            show_grid_fingers = not show_grid_fingers
            print(f"Grid finger numbers: {'ON' if show_grid_fingers else 'OFF'}")
        elif key == ord('n'):
            current_chord_idx = (current_chord_idx + 1) % len(available_chords)
            tutor.set_chord(available_chords[current_chord_idx])
            chord_completed = False
            print(f"Now teaching: {CHORD_LIBRARY[available_chords[current_chord_idx]]['name']}")
        elif key == ord('p'):
            current_chord_idx = (current_chord_idx - 1) % len(available_chords)
            tutor.set_chord(available_chords[current_chord_idx])
            chord_completed = False
            print(f"Now teaching: {CHORD_LIBRARY[available_chords[current_chord_idx]]['name']}")
        elif key == ord(' '):  # Spacebar to continue after checkmark
            if chord_completed:
                current_chord_idx = (current_chord_idx + 1) % len(available_chords)
                tutor.set_chord(available_chords[current_chord_idx])
                chord_completed = False
                print(f"Now teaching: {CHORD_LIBRARY[available_chords[current_chord_idx]]['name']}")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✓ AI Guitar Tutor closed")


if __name__ == "__main__":
    main()