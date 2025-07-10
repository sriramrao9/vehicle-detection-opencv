import cv2
import numpy as np

# Load Haar cascade
car_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_car.xml')
# Check if file loaded correctly
if car_cascade.empty():
    print("🚫 ERROR: Failed to load Haar cascade file. Check the file name and path.")
    exit()


# Capture video (0 = webcam, or replace with 'videos/sample.mp4')
cap = cv2.VideoCapture(0)

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    roi = frame[200:480, :]  # Bottom half = road area

    # Background subtraction
    fg_mask = bg_subtractor.apply(roi)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Clean noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Contour detection
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 400:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Haar cascade vehicle detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

    frame[200:480, :] = roi
    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
