import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import pymongo
from datetime import datetime

model = load_model('moroccan_sign_language_model.h5') 

input_shape = model.input_shape[1:]
print(f"Model expects input shape: {input_shape}")

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["sign_language_db"]
detection_history = db["detection_history"]

class_names = [
    'aleff', 'ain', 'al', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain',
    'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra',
    'saad', 'seen', 'sheen', 'ta', 'taa', 'thal', 'thaa', 'toot', 'waw', 'ya',
    'yaa', 'zay'
]
def extract_hand_region(frame):
    """Extracts the hand region as a cropped image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 3000:
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            padding = 20
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)

            hand_crop = frame[y_start:y_end, x_start:x_end]

            return True, hand_crop, frame, (x, y, w, h)

    return False, None, frame, None



def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    curr_time = 0
    prediction_buffer = []
    buffer_size = 5
    confidence_threshold = 0.70
    last_recorded_sign = None
    prediction_cooldown = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        hand_detected, hand_image, frame, hand_rect = extract_hand_region(frame)

        prediction_text = "No hand detected"
        confidence = 0

        if hand_detected:
            resized_image = cv2.resize(hand_image, (input_shape[0], input_shape[1]))
            normalized_image = resized_image / 255.0
            input_data = np.expand_dims(normalized_image, axis=0)

            predictions = model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[0][predicted_class_index]
            predicted_class = class_names[predicted_class_index]


            prediction_buffer.append((predicted_class, confidence))
            if len(prediction_buffer) > buffer_size:
                prediction_buffer.pop(0)

            class_counts = {}
            for pred, conf in prediction_buffer:
                if conf >= confidence_threshold:
                    class_counts[pred] = class_counts.get(pred, 0) + 1

            most_frequent = max(class_counts, key=class_counts.get, default=None)
            if most_frequent and class_counts[most_frequent] >= buffer_size // 2:
                prediction_text = f"Sign: {most_frequent} ({confidence:.2f})"

                if prediction_cooldown <= 0 and (last_recorded_sign != most_frequent) and confidence >= confidence_threshold:
                    detection_history.insert_one({
                        "sign": most_frequent,
                        "confidence": float(confidence),
                        "timestamp": datetime.now()
                    })
                    last_recorded_sign = most_frequent
                    prediction_cooldown = 15
            else:
                prediction_text = "Analyzing..."



            if hand_rect:
                x, y, w, h = hand_rect
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if prediction_cooldown > 0:
            prediction_cooldown -= 1

        cv2.putText(frame, prediction_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Moroccan Sign Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()