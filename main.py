import cv2
import mediapipe as mp
import random
import time

# --- Bubble Class for Hotpot effect ---
class Bubble:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = random.randint(5, 20)
        self.life = 1.0  # starts fully visible

    def update(self):
        self.y -= 2  # rise
        self.life -= 0.02  # fade out

def draw_bubbles(frame, bubbles):
    for b in bubbles[:]:
        if b.life <= 0:
            bubbles.remove(b)
            continue
        color = (0, int(165 * b.life), int(255 * b.life))  # orange to light
        cv2.circle(frame, (int(b.x), int(b.y)), int(b.radius), color, -1)
        b.update()

# Initialize MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start webcam
cap = cv2.VideoCapture(0)
bubbles = []
cv2.startWindowThread()
cv2.namedWindow('QueuePlay Wall Prototype', cv2.WINDOW_NORMAL)
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame (OpenCV) to RGB (MediaPipe)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Convert back to BGR to display with OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        right_hand = None
        nose = None

        # Draw landmarks if detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            except IndexError:
                pass  # skip frame if something is missing

            if right_hand and nose and right_hand.y < nose.y:
                bubbles.append(Bubble(int(right_hand.x * frame.shape[1]),
                                      int(right_hand.y * frame.shape[0])))

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            # Optional: highlight the wrists
            h, w, _ = image.shape
            left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            lx, ly = int(left.x * w), int(left.y * h)
            rx, ry = int(right.x * w), int(right.y * h)
            cv2.circle(image, (lx, ly), 15, (0, 255, 255), -1)
            cv2.circle(image, (rx, ry), 15, (0, 255, 255), -1)
        
        draw_bubbles(image, bubbles)

        # Display
        cv2.imshow('QueuePlay Wall Prototype', cv2.flip(image, 1))  # mirror view

        # Exit when ESC pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
