import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(
    staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.8, minTrackCon=0.5
)

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# Initialize video capture
cap = cv2.VideoCapture(0)

# Reduce the resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Parameters
wrist_landmark_id = mp_hands.HandLandmark.WRIST
index_landmark_id = mp_hands.HandLandmark.INDEX_FINGER_TIP
middle_landmark_id = mp_hands.HandLandmark.MIDDLE_FINGER_TIP

# Zone and box properties (adjust values as needed)
BASE_ZONE_HEIGHT = 120  # Adjusted to accommodate fingers
BASE_ZONE_WIDTH = 200
ZONE_COLOR = (0, 255, 0)  # Green color
BOX_OFFSET_X = 15  # Offset to bring boxes closer to hand (adjust as needed)
BOX_OFFSET_Y = 5   # Offset for fine-tuning
BASE_BOX_WIDTH = 60
BASE_BOX_HEIGHT = 60

# Store previous hand positions
prev_hand_y = None

SCROLL_DEBOUNCE_FRAMES = 5  # Number of frames the hand must stay in the zone to trigger scroll

# Deques to store recent hand positions
top_zone_frames = deque(maxlen=SCROLL_DEBOUNCE_FRAMES)
bottom_zone_frames = deque(maxlen=SCROLL_DEBOUNCE_FRAMES)
top_zone_frames_slow = deque(maxlen=SCROLL_DEBOUNCE_FRAMES)
bottom_zone_frames_slow = deque(maxlen=SCROLL_DEBOUNCE_FRAMES)

scrolling_paused = False

def scale_zone_dimensions(wrist_z, base_value):
    scale_factor = max(0.3, min(0.8, 0.6/ (wrist_z + 0.1)))  # Scale factor between 0.5 and 2.0
    return int(base_value * scale_factor)

while True:
    success, img = cap.read()
    if not success: 
        break
    handsp ,img = detector.findHands(img, draw=True, flipType=True)
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural (selfie-view) visualization
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_frame)
    if handsp:
        hand1=handsp[0]
        if hand1["type"]=="Right":
         pos = detector.fingersUp(hand1)

         if pos == [0, 0, 0, 0, 0]:
                scrolling_paused = True
         elif pos == [1, 1, 1, 1, 1]:
            scrolling_paused = False

         if not scrolling_paused and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[index_landmark_id]
                index_x = int(index_tip.x * frame.shape[1])
                index_y = int(index_tip.y * frame.shape[0])
                middle_tip = hand_landmarks.landmark[middle_landmark_id]
                middle_x = int(middle_tip.x * frame.shape[1])
                middle_y = int(middle_tip.y * frame.shape[0])

                # Extract landmark coordinates for wrist
                wrist_landmark = hand_landmarks.landmark[wrist_landmark_id]
                wrist_x = int(wrist_landmark.x * frame.shape[1])
                wrist_y = int(wrist_landmark.y * frame.shape[0])
                wrist_z = wrist_landmark.z
                
                ZONE_HEIGHT = scale_zone_dimensions(wrist_z, BASE_ZONE_HEIGHT)
                ZONE_WIDTH = scale_zone_dimensions(wrist_z, BASE_ZONE_WIDTH)
                BOX_WIDTH = scale_zone_dimensions(wrist_z, BASE_BOX_WIDTH)
                BOX_HEIGHT = scale_zone_dimensions(wrist_z, BASE_BOX_HEIGHT)
                y_offset_up = int(scale_zone_dimensions(wrist_y / frame.shape[0], BASE_ZONE_HEIGHT))+30
                y_offset_down = int(scale_zone_dimensions(wrist_y / frame.shape[0], BASE_ZONE_HEIGHT))-2*BOX_HEIGHT
                x_offset = int(scale_zone_dimensions(wrist_y / frame.shape[0], BASE_ZONE_HEIGHT))+2*BOX_HEIGHT                # Calculate positions for scroll zones and boxes
                top_zone_top_left = (wrist_x-x_offset - ZONE_WIDTH, wrist_y - y_offset_up - 2 * BOX_HEIGHT)
                top_zone_bottom_right = (wrist_x-x_offset + ZONE_WIDTH, wrist_y - y_offset_up)
                bottom_zone_top_left = (wrist_x-x_offset - ZONE_WIDTH, wrist_y + y_offset_down)
                bottom_zone_bottom_right = (wrist_x-x_offset + ZONE_WIDTH, wrist_y + y_offset_down + 2 * BOX_HEIGHT)

                # Draw scroll zones on the frame
                cv2.rectangle(frame, top_zone_top_left, top_zone_bottom_right, ZONE_COLOR, 2)
                cv2.rectangle(frame, bottom_zone_top_left, bottom_zone_bottom_right, ZONE_COLOR, 2)

                # Check if hand is within the scroll zones
                hand_in_top_zone_slow = (top_zone_top_left[1] < middle_y < top_zone_bottom_right[1])
                hand_in_bottom_zone_slow = (bottom_zone_top_left[1] < middle_y < bottom_zone_bottom_right[1])
                hand_in_top_zone_fast = (top_zone_top_left[1] <(index_y and middle_y) < top_zone_bottom_right[1])
                hand_in_bottom_zone_fast = (bottom_zone_top_left[1] < (index_y and middle_y) < bottom_zone_bottom_right[1])

                # Scroll based on hand position relative to scroll zones
                top_zone_frames.append(hand_in_top_zone_fast)
                bottom_zone_frames.append(hand_in_bottom_zone_fast)
                top_zone_frames_slow.append(hand_in_top_zone_slow)
                bottom_zone_frames_slow.append(hand_in_bottom_zone_slow)

                if sum(top_zone_frames) == SCROLL_DEBOUNCE_FRAMES:
                    pyautogui.scroll(250)  # Scroll up
                    print("Scrolling up fast")
                    top_zone_frames.clear()  # Clear the deque after scrolling
                elif sum(bottom_zone_frames) == SCROLL_DEBOUNCE_FRAMES:
                    pyautogui.scroll(-250)  # Scroll down
                    print("Scrolling down fast")
                    bottom_zone_frames.clear()

                if sum(top_zone_frames_slow) == SCROLL_DEBOUNCE_FRAMES:
                    pyautogui.scroll(100)  # Scroll up
                    print("Scrolling up slow")
                    top_zone_frames.clear()  # Clear the deque after scrolling
                elif sum(bottom_zone_frames_slow) == SCROLL_DEBOUNCE_FRAMES:
                    pyautogui.scroll(-100)  # Scroll down
                    print("Scrolling down slow")
                    bottom_zone_frames.clear()
                    
                

    # Display the frame
    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
