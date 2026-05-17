import cv2
import numpy as np
import os
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Constants
DATA_PATH = os.path.join('Feature_Extraction')
actions = np.array(['hello', 'thanks', 'iloveyou'])
number_sequences = 30
sequence_length = 30

def create_folders():
    """Create folders for data collection."""
    for action in actions:
        for sequence in range(number_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def collect_data():
    """Capture frames from webcam and save landmarks as numpy arrays."""
    create_folders()
    
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(number_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Couldn't read frame from webcam.")
                        continue
                        
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    if frame_num == 0:
                        cv2.putText(image, 'START NEW ACTION', (120, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Action Detection', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Action Detection', image)
                    
                    keypoints = extract_keypoints(results)
                    keypoint_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(keypoint_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data()
