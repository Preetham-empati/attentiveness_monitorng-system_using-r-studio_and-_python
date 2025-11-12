import cv2
import csv
from datetime import datetime
import time
import mediapipe as mp
import math
from scipy.spatial import distance as dist
import numpy as np

# --- Constants & Calibration ---
# --- TUNE THESE VALUES ---
# 3D "Focus Zone" Thresholds (in degrees)
FOCUS_PITCH_THRESHOLD = 25  # How far "down" they can look
FOCUS_YAW_THRESHOLD = 30    # How far "left/right" they can look

# Sleepy Thresholds
EAR_THRESHOLD = 0.25      # EAR value below which eyes are "closed"
EAR_CONSEC_FRAMES = 30    # Consecutive frames to be "Sleepy"

# Tracking
MAX_CENTROID_DISTANCE = 100 # Max pixels a student can move between frames

# --- NEW: Performance Tuning ---
HAND_DETECTION_SKIP_FRAMES = 10 # Run hand detection only every 10 frames

# --- NEW: Granular Score Weights (Updated for 3D Pose) ---
SCORE_WEIGHTS = {
    "Attentive": 1.0,
    "Looking Down": 0.6,
    "Looking Away": 0.5,
    "Face Covered": 0.3,
    "Sleepy": 0.0
}

CSV_FILE_NAME = "classroom_log.csv"

# --- MediaPipe Initialization ---
print("[INFO] Loading MediaPipe models...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Define Eye Landmark Indices ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- NEW: 3D Head Pose Model Points ---
model_points_3d = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# --- Helper Functions ---
def calculate_ear(eye_landmarks, face_landmarks, frame_shape):
    h, w = frame_shape
    coords = []
    for idx in eye_landmarks:
        lm = face_landmarks.landmark[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))
    A = math.dist(coords[1], coords[5])
    B = math.dist(coords[2], coords[4])
    C = math.dist(coords[0], coords[3])
    if C == 0: return 0.3
    ear = (A + B) / (2.0 * C)
    return ear

def check_overlap(bbox1, bbox2):
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2]: return False
    if bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]: return False
    return True

def get_bbox_and_centroid(landmarks, h, w):
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    bbox = [x_min, y_min, x_max, y_max]
    centroid = ((x_min + x_max) // 2, (y_min + y_max) // 2)
    return bbox, centroid

# --- NEW: 3D Head Pose Function ---
def get_3d_head_pose(face_landmarks, h, w):
    # Get 2D image points from MediaPipe
    image_points_2d = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),       # Nose
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),    # Chin
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),    # Left eye
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),      # Right eye
        (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h),    # Left mouth
        (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h)       # Right mouth
    ], dtype=np.float64)

    # --- Camera Intrinsics (Assume a standard camera) ---
    focal_length = w
    cam_center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, cam_center[0]],
        [0, focal_length, cam_center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1)) # Assume no distortion

    # --- SolvePnP: Find 3D rotation ---
    (success, rvec, tvec) = cv2.solvePnP(model_points_3d, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Project 3D axis for visualization
    axis_3d = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200]]).reshape(-1, 3)
    (axis_2d, _) = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # --- Get Euler Angles (Yaw, Pitch, Roll) ---
    R, _ = cv2.Rodrigues(rvec)
    
    # Unpack 7 values, the angles are the 7th item
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((R, tvec)))
    
    # --- THIS IS THE FIX ---
    # Extract the scalar value from the 1-element array
    yaw = angles[1][0]
    pitch = angles[0][0]
    roll = angles[2][0]
    
    return pitch, yaw, roll, axis_2d.reshape(3, 2).astype(int), (int(image_points_2d[0][0]), int(image_points_2d[0][1]))

# --- CSV File Setup ---
print(f"[INFO] Opening CSV file: {CSV_FILE_NAME}")
csv_file = open(CSV_FILE_NAME, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "roll_no", "status", "attentiveness_score"])

# --- Video Stream Initialization ---
print("[INFO] Starting video stream...")
CAMERA_INDEX = 0 # Your DroidCam
cap = cv2.VideoCapture(CAMERA_INDEX)
time.sleep(1.0)

# --- Tracking & Performance Variables ---
previous_objects = {}  # {objectID: (centroid, sleep_counter)}
next_objectID = 0
frame_counter = 0
hand_bboxes = [] # Keep hand boxes until next check

# --- Main Loop with MediaPipe ---
with mp_face_mesh.FaceMesh(
    max_num_faces=60,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(
    max_num_hands=10,
    min_detection_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_counter += 1
        
        # --- 1. Hand Detection (Optimized with Frame Skipping) ---
        if frame_counter % HAND_DETECTION_SKIP_FRAMES == 0:
            hand_bboxes = [] # Clear old boxes
            hand_results = hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    hand_bboxes.append([int(min(x_coords) * w), int(min(y_coords) * h),
                                        int(max(x_coords) * w), int(max(y_coords) * h)])

        # --- 2. Face Detection & Tracking Setup ---
        face_results = face_mesh.process(frame_rgb)
        current_frame_objects = [] # (centroid, face_landmarks, bbox)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                bbox, centroid = get_bbox_and_centroid(face_landmarks.landmark, h, w)
                current_frame_objects.append((centroid, face_landmarks, bbox))

        # --- 3. Persistent Tracking Logic ---
        current_objects = {} # {objectID: (centroid, sleep_counter)}
        
        for (new_centroid, face_landmarks, bbox) in current_frame_objects:
            best_match_id, min_dist = -1, MAX_CENTROID_DISTANCE
            for objectID, (prev_centroid, _) in previous_objects.items():
                distance = dist.euclidean(prev_centroid, new_centroid)
                if distance < min_dist:
                    min_dist, best_match_id = distance, objectID
            
            if best_match_id != -1:
                objectID = best_match_id
                sleep_counter = previous_objects.get(objectID, (0, 0))[1] # Safer get
            else:
                objectID, next_objectID = next_objectID, next_objectID + 1
                sleep_counter = 0

            # --- 4. Get All Status Metrics ---
            is_covered = any(check_overlap(bbox, hand_bbox) for hand_bbox in hand_bboxes)
            
            # --- 5. NEW: 3D Head Pose ---
            try:
                pitch, yaw, roll, pose_axis, nose_tip = get_3d_head_pose(face_landmarks, h, w)
            except Exception as e:
                # print(f"Error in solvePnP: {e}")
                continue # Skip this face if 3D pose fails

            # EAR
            left_ear = calculate_ear(LEFT_EYE_INDICES, face_landmarks, (h, w))
            right_ear = calculate_ear(RIGHT_EYE_INDICES, face_landmarks, (h, w))
            ear = (left_ear + right_ear) / 2.0
            
            # --- 6. Final Status & Score Logic ---
            status = "Attentive"
            if pitch > FOCUS_PITCH_THRESHOLD:
                status = "Looking Down"
            elif abs(yaw) > FOCUS_YAW_THRESHOLD:
                status = "Looking Away"
            
            if ear < EAR_THRESHOLD:
                sleep_counter += 1
                if sleep_counter >= EAR_CONSEC_FRAMES:
                    status = "Sleepy"
            else:
                sleep_counter = 0
            
            if is_covered: status = "Face Covered"

            attentiveness_score = SCORE_WEIGHTS.get(status, 0.0)
            
            # --- 7. Log Data ---
            roll_no = f"student_{objectID}"
            timestamp = datetime.now().isoformat()
            csv_writer.writerow([timestamp, roll_no, status, attentiveness_score])
            
            # --- 8. Update state for next frame ---
            current_objects[objectID] = (new_centroid, sleep_counter)

            # --- 9. Visualization ---
            color = (0, 255, 0) if status == "Attentive" else (0, 0, 255)
            (x_min, y_min, x_max, y_max) = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{roll_no}: {status}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw 3D Pose Axis
            cv2.line(frame, nose_tip, tuple(pose_axis[0]), (0, 0, 255), 3) # Y-axis (Pitch)
            cv2.line(frame, nose_tip, tuple(pose_axis[1]), (0, 255, 0), 3) # X-axis (Yaw)
            cv2.line(frame, nose_tip, tuple(pose_axis[2]), (255, 0, 0), 3) # Z-axis (Roll)
            
            cv2.putText(frame, f"P: {pitch:.0f} Y: {yaw:.0f}", (x_min, y_max + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        previous_objects = current_objects.copy()
        cv2.imshow("Classroom Attentiveness Monitor", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

# --- Cleanup ---
print("[INFO] Cleaning up...")
csv_file.close()
cap.release()
cv2.destroyAllWindows()