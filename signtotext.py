import cv2
import mediapipe as mp

# Initialize MediaPipe components for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create a Hands object with max_num_hands=1
hands = mp_hands.Hands(max_num_hands=1)

# Define a mapping of gestures to text
def gesture_to_text(l):
    # Get the coordinates of the key landmarks
    t_tip = l[4]
    t_ip = l[3]
    i_tip = l[8]
    m_tip = l[12]
    r_tip = l[16]
    p_tip = l[20]
    
    # Example gesture detection logic
    # Thumb Up
    if t_tip.y < t_ip.y and i_tip.y < t_tip.y:
        return "Up 👍"
    
    # Thumb Down
    if t_tip.y > t_ip.y and i_tip.y > t_tip.y:
        return "Down 👎"
    
    # OK Gesture (O shape with thumb and index finger)
    if abs(t_tip.y - i_tip.y) < 0.02 and t_tip.x < i_tip.x:
        return "OK 👌"
    
    # Not OK Gesture (Fist or closed hand)
    if all([abs(t_tip.y - other_tip.y) > 0.1 for other_tip in [i_tip, m_tip, r_tip, p_tip]]):
        return "Not OK 🚫"
    
    # Left Gesture (Hand open with fingers pointing left)
    if t_tip.x < i_tip.x < m_tip.x < r_tip.x < p_tip.x:
        return "Left ⬅"
    
    # Right Gesture (Hand open with fingers pointing right)
    if t_tip.x > i_tip.x > m_tip.x > r_tip.x > p_tip.x:
        return "Right ▶"
    
    # Stop Gesture (Hand open with palm facing outwards)
    if all([abs(t_tip.y - other_tip.y) < 0.1 for other_tip in [i_tip, m_tip, r_tip, p_tip]]):
        return "Stop ✋"
    
    # Hi Gesture (Hand with fingers spread out, palm facing camera)
    if abs(t_tip.x - i_tip.x) > 0.1 and abs(i_tip.y - t_tip.y) < 0.05:
        return "Hi 👋"
    
    # Bye Gesture (Waving hand, palm moving left and right)
    if t_tip.y < i_tip.y < m_tip.y < r_tip.y < p_tip.y:
        return "Bye 👋"
    
    # Default unknown gesture
    return "Unknown Gesture"

# Access the laptop camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks and display detected gesture text
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Convert detected hand landmarks to text
            gesture_text = gesture_to_text(hand_landmarks.landmark)
            
            # Display the gesture text on the frame
            cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with overlayed gesture text
    cv2.imshow('Sign Language to Text Converter', frame)
    
    # Exit the application when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
