import cv2
import os
import numpy as np

# Folder with training photos
dataPath = './data'
peopleList = os.listdir(dataPath)
print('List of people: ', peopleList)

# Initialize lists to store labels and face data
labels = []
facesData = []
label = 0

# Use loops to read all the images from the training set
print("Reading the images...")
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir

    # Search for all files in each of the folders
    for fileName in os.listdir(personPath):
        print('Faces:', nameDir + '/' + fileName)

        # Add the label and the face image to their respective arrays
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))

    label = label + 1

# Create an LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer.create()

print("Training the model... ")
# Train the recognizer with face data and corresponding labels
face_recognizer.train(facesData, np.array(labels))

# Save the trained model to a file
face_recognizer.save('model.xml')
print("Model Saved")
