import cv2
from camera import getcamera

# -------------------------------------------------------------------------
# Write your code here:
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

ret, frame = cap.read()

cv2.imshow('frame', frame)

cv2.waitKey(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------------------------------------------------
# Close the camera and windows - do not delete these lines
# Leave these lines at the bottom
cv2.destroyAllWindows()
cap.release()
