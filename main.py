import cv2
import mediapipe as mp

# Initialize MediaPipe Hand and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand detection with MediaPipe
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Gesture recognition function
def recognize_gesture(landmarks):
    # Thumb is up if landmark 4 is above landmark 3 and 2 (relative to y-axis)
    thumb_up = landmarks[4].y < landmarks[3].y < landmarks[2].y
    # Peace sign: Index and middle fingers extended while ring and pinky fingers are down
    peace_sign = (landmarks[8].y < landmarks[6].y) and (landmarks[12].y < landmarks[10].y) and \
                 (landmarks[16].y > landmarks[14].y) and (landmarks[20].y > landmarks[18].y)
    
    if thumb_up:
        return "Thumbs Up"
    elif peace_sign:
        return "Peace Sign"
    else:
        return "Unknown Gesture"

while cap.isOpened():
    success, frame = cap.read()  # Capture frame from webcam
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks and recognize gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Recognize gestures
            gesture = recognize_gesture(hand_landmarks.landmark)
            
            # Display the gesture on the screen
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame with landmarks and gesture text
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
