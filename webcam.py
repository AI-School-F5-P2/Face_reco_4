#
import cv2
import uuid   # Universally Unique Identifier
from dotenv import load_dotenv
import os

# loading of .env variables for the path
load_dotenv()

#Connection to Webcam
cap = cv2.VideoCapture(0)

# Loop to capture frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      exit()

    # Image size change with capture of the size ratio
    width, height = 400, 300
    resized_frame = cv2.resize(frame, (width, height))

    frame = resized_frame[50:50 + 218, 120:120 + 178, :]

    #Collect positives
    if cv2.waitKey(1) & 0XFF  == ord('p'):
        # Create the unique file path
        positive = os.getenv('Pos_Path')
        positive_name = os.path.join(positive, '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(positive_name, frame)

    #Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path
        anchor = os.getenv('Anc_Path')
        anchor_name = os.path.join(anchor,'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(anchor_name, frame)

    # Show images back to screen
    cv2.imshow('Collection Image', frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
