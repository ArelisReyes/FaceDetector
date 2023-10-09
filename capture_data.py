import cv2
import os
import imutils
from camera import getcamera

# Create a person's folder:
print('Enter your name: ')
personName = input()
dataPath = './data'
personPath = dataPath + '/' + personName

# Display the action to be taken depending on the entered name
if os.path.exists(personPath):
    print('Person already registered, overwriting data...')
else:
    os.makedirs(personPath)
    print('New person, capturing data...')

camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier("faces.xml")

counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=640)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(120, 120),
                                         maxSize=(1000, 1000))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        auxFrame = frame.copy()

        face = auxFrame[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(personPath + '/face_{}.jpg'.format(counter), face)
        print('face_{}.jpg'.format(counter) + ' saved')

        counter = counter + 1

    cv2.imshow('frame', frame)

    if counter >= 300 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera and windows - do not delete these lines
cv2.destroyAllWindows()
cap.release()
