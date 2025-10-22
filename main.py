import cv2
import mediapipe as mp
import random

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

# --- Overlay helper ---
def overlay_image(background, overlay, x, y):
    if overlay is None or overlay.shape[2] < 4:
        return  # skip if overlay missing or no transparency channel
    h, w = overlay.shape[:2]
    if y < 0 or x < 0 or y + h > background.shape[0] or x + w > background.shape[1]:
        return  # skip if overlay goes out of frame
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] + (1 - alpha) * background[y:y+h, x:x+w, c]
        )

# Load hotpot top view image
hotpot_img = cv2.imread("hotpot_topview.png", cv2.IMREAD_UNCHANGED)
if hotpot_img is None:
    print("Error: Could not load 'hotpot_topview.png'.")
else:
    print(f"Hotpot image loaded: {hotpot_img.shape}")
    # Convert to 4 channels if it's only 3
    if hotpot_img.shape[2] == 3:
        hotpot_img = cv2.cvtColor(hotpot_img, cv2.COLOR_BGR2BGRA)
        print("Converted hotpot image to 4 channels (BGRA).")

# Initialize MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start webcam
cap = cv2.VideoCapture(0)
bubbles = []
cv2.namedWindow('QueuePlay Wall Prototype', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera feed not available.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            h, w, _ = image.shape
            right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Generate bubbles if hand above nose
            if right_hand.y < nose.y:
                bubbles.append(Bubble(int(right_hand.x * w), int(right_hand.y * h)))

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
            
            # Overlay hotpot image at torso center
            if hotpot_img is not None:
                cx = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                cy = int((left_shoulder.y + right_shoulder.y) / 2 * h)

                pot_size = 200
                pot_resized = cv2.resize(hotpot_img, (pot_size, pot_size))

                overlay_image(image, pot_resized, cx - pot_size // 2, cy - pot_size // 2)

                # Debug red dot
                cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)




        draw_bubbles(image, bubbles)
        cv2.imshow('QueuePlay Wall Prototype', cv2.flip(image, 1))

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
