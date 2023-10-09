import cv2

# Open the camera:
cap = cv2.VideoCapture(0)

# Create a face classifier
faceClassif = cv2.CascadeClassifier('faces.xml')

# Create an eye classifier
eyeClassif = cv2.CascadeClassifier('eyes.xml')

# Use the camera until the 'q' key is pressed
while True:
    # Capture a frame and display it on the screen
    ret, frame = cap.read()

    # Check if an image exists
    if not ret:
        break

    # Create a grayscale image from the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(120, 120),
                                         maxSize=(1000, 1000))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of interest to search for eyes within the face area
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eyeClassif.detectMultiScale(roi_gray,
                                           scaleFactor=1.1,
                                           minNeighbors=10,
                                           minSize=(50, 50),
                                           maxSize=(300, 300))

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera and windows
cv2.destroyAllWindows()
cap.release()
