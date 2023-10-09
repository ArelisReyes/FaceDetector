import cv2
import os
from camera import getcamera

# Code Structure - Find where to start your code, do not modify the rest

# Find saved people's names
dataPath = './data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# Creating the model and reading the model
# noinspection PyUnresolvedReferences
face_recognizer = cv2.face_LBPHFaceRecognizer_create()
face_recognizer.read('model.xml')

# Create face classifier
faceClassif = cv2.CascadeClassifier('faces.xml')

# Open the camera:
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# Use the camera until the 'q' key is pressed
while True:
    # Take a photo and display it on the screen
    ret, frame = cap.read()

    # Check if an image exists
    if not ret:
        break

    # Create a grayscale image from the photo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a copy of the grayscale image
    auxFrame = gray.copy()

    # Use the detector on the grayscale image
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # -------------------------------------------------------------------------
    # Write your code here:
    for (x, y, w, h) in faces:

        face = auxFrame[y:y + h, x:x + w]

        face = cv2.resize(face, (150, 150))

        result = face_recognizer.predict(face)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1, (255, 255, 0))

        # Display results on screen
        if result[1] < 75:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Unknown', (x, y - 20), 2, 1, (0, 0, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # -------------------------------------------------------------------------
    # Do not modify the code below this line:
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------------------------------------------------
# Close the camera and windows - do not delete these lines
# Keep these lines at the bottom
cv2.destroyAllWindows()
cap.release()
