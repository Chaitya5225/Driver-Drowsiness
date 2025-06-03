import dlib
import sys
import cv2
import time     
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import pygame
import queue

# ------------------------ Configuration & Initialization ------------------------
FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460
THRESH = 0.27
MODEL_PATH = "models/shape_predictor_70_face_landmarks.dat"
ALARM_SOUND_PATH = "alarm.wav"

# Dlib detectors and predictors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Eye landmark indices
leftEyeIndices = [36, 37, 38, 39, 40, 41]
rightEyeIndices = [42, 43, 44, 45, 46, 47]

# State variables
blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.1
drowsyTime = 1
ALARM_ON = False
threadStatusQ = queue.Queue()

# Gamma correction table
GAMMA = 1.5
invGamma = 1.0 / GAMMA
gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")

# ------------------------------ Helper Functions --------------------------------
def gamma_correction(image):
    return cv2.LUT(image, gamma_table)

def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def soundAlert(sound_path, status_queue):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.set_volume(1.0)
    pygame.mixer.music.play(-1)  # Loop indefinitely

    while True:
        try:
            stop_signal = status_queue.get(timeout=0.1)
            if stop_signal:
                pygame.mixer.music.stop()
                break
        except queue.Empty:
            continue

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def checkEyeStatus(landmarks):
    leftEye = np.array([landmarks[i] for i in leftEyeIndices])
    rightEye = np.array([landmarks[i] for i in rightEyeIndices])
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return 1 if (leftEAR + rightEAR) / 2.0 >= THRESH else 0

def checkBlinkStatus(eyeStatus, falseBlinkLimit, drowsyLimit):
    global state, blinkCount, drowsy
    if state <= falseBlinkLimit:
        state = 0 if eyeStatus else state + 1
    elif state < drowsyLimit:
        if eyeStatus:
            blinkCount += 1
            state = 0
        else:
            state += 1
    else:
        drowsy = 1
        if eyeStatus:
            blinkCount += 1
            state = 0

def getLandmarks(image):
    small_image = cv2.resize(image, None, fx=1.0 / FACE_DOWNSAMPLE_RATIO, fy=1.0 / FACE_DOWNSAMPLE_RATIO)
    rects = detector(small_image, 0)
    if not rects:
        return None
    rect = rects[0]
    scaled_rect = dlib.rectangle(int(rect.left() * FACE_DOWNSAMPLE_RATIO),
                                 int(rect.top() * FACE_DOWNSAMPLE_RATIO),
                                 int(rect.right() * FACE_DOWNSAMPLE_RATIO),
                                 int(rect.bottom() * FACE_DOWNSAMPLE_RATIO))
    shape = predictor(image, scaled_rect)
    return [(p.x, p.y) for p in shape.parts()]

def drawEyeLandmarks(frame, landmarks):
    for i in leftEyeIndices + rightEyeIndices:
        cv2.circle(frame, landmarks[i], 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

def process_frame(frame):
    height = frame.shape[0]
    scale = height / RESIZE_HEIGHT
    resized_frame = cv2.resize(frame, None, fx=1 / scale, fy=1 / scale)
    adjusted = histogram_equalization(resized_frame)
    return resized_frame, adjusted

# ---------------------------------- Main Loop -----------------------------------
def main():
    global ALARM_ON, state, drowsy, blinkCount
    capture = cv2.VideoCapture(0)
    
    for _ in range(10):
        capture.read()

    print("Calibration in progress...")
    total_time, valid_frames = 0.0, 0
    dummy_frames = 100

    while valid_frames < dummy_frames:
        ret, frame = capture.read()
        if not ret:
            continue
        resized_frame, adjusted = process_frame(frame)
        start = time.time()
        landmarks = getLandmarks(adjusted)
        elapsed = time.time() - start
        if landmarks is None:
            cv2.putText(resized_frame, 
                        "Unable to detect face. Check lighting.", 
                        (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Calibration", resized_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                capture.release()
                cv2.destroyAllWindows()
                sys.exit()
            continue
        total_time += elapsed
        valid_frames += 1

    spf = total_time / dummy_frames
    drowsyLimit = drowsyTime / spf
    falseBlinkLimit = blinkTime / spf
    print(f"Calibration done. SPF: {spf:.3f}, Drowsy limit: {drowsyLimit:.2f}, Blink limit: {falseBlinkLimit:.2f}")
    cv2.imshow("Calibration", resized_frame)
    cv2.waitKey(1)
    cv2.destroyWindow("Calibration")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_writer = cv2.VideoWriter('output-low-light-optimized.avi', 
                                 fourcc, 15, (resized_frame.shape[1], resized_frame.shape[0]))

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        resized_frame, adjusted = process_frame(frame)
        landmarks = getLandmarks(adjusted)
        if landmarks is None:
            cv2.putText(resized_frame, 
                        "Unable to detect face. Check lighting.", 
                        (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Drowsiness Detection", resized_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        eyeStatus = checkEyeStatus(landmarks)
        drowsy = 0
        checkBlinkStatus(eyeStatus, falseBlinkLimit, drowsyLimit)
        drawEyeLandmarks(resized_frame, landmarks)

        if drowsy:
            cv2.putText(resized_frame, "!!! DROWSINESS ALERT !!!", (70, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            if not ALARM_ON:
                ALARM_ON = True
                threadStatusQ.put(False)
                Thread(target=soundAlert, args=(ALARM_SOUND_PATH, threadStatusQ), daemon=True).start()
        else:
            cv2.putText(resized_frame, f"Blinks: {blinkCount}", (460, 80),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            if ALARM_ON:
                threadStatusQ.put(True)
                ALARM_ON = False

        cv2.imshow("Drowsiness Detection", resized_frame)
        out_writer.write(resized_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            state = drowsy = blinkCount = 0
            ALARM_ON = False
            threadStatusQ.put(True)
        elif key == 27:
            break

    capture.release()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
