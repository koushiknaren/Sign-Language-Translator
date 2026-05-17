import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Constants
actions = np.array(['hello', 'thanks', 'iloveyou'])
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def prob_viz(res, actions, input_frame, colors):
    """Visualize probabilities of predicted actions."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def real_time_test():
    """Run real time testing using webcam."""
    try:
        model = load_model('lstm_model.h5')
    except Exception as e:
        print("Error loading model. Make sure you have trained the model using train_model.py first.")
        print(f"Details: {e}")
        return

    sequence = []
    sentence = []
    threshold = 0.8
    
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame from webcam.")
                break
                
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                
                # Logic to display sentence
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Real Time Translator', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time_test()
