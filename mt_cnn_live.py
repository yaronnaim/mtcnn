import mtcnn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle, Circle
import cv2
import numpy as np


def plot_bounding_box(image, result, scale=1,
                      show_facial_landmarks = False,
                      show_confidence = False):
    """

    :param image:  Original size image
    :param result: MTCNN output, Dict keys 'box' and 'keypoints'
    :param scale:  Scale down factor betwwen original to MTCNN image
    :param show_facial_landmarks: Bool to indicate if facial landmark should be
    plotted
    :param show_confidence: Bool to indicate if confidence should be
    plotted
    :return: None
    """
    x, y, w, h = [i*scale for i in result['box']]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if show_facial_landmarks is True:
        for key, value in result['keypoints'].items():
            cv2.circle(image, (value[0]*scale, value[1]*scale),
                                radius=2,
                                color=(255, 0, 0),
                                thickness=2)
    if show_confidence is True:
        confidence_string = str(round(result['confidence'], 3))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # fontScale
        fontScale = 0.5

        # Blue color in BGR
        color = (0, 255, 0)

        # Line thickness of 2 px
        thickness = 1

        cv2.putText(image, confidence_string, (x, y-2), font, fontScale,
                    color, thickness)
    return


    # Print MTCC version
print('Multi-Task Cascaded Convolutional Neural Network, version = ',
                    mtcnn.__version__)

# create the detector, using default weights
detector = MTCNN()
# detect faces in the image

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    (width, height, color) = img.shape
    # print(width, height, color)

    # Resize image by decimation factor
    decimation = 10
    height2 = np.int(height/decimation)
    width2 = np.int(width/decimation)

    # Resize and convert to RGB
    RGBimageResized = cv2.cvtColor(cv2.resize(img, (height2, width2)),
                                  cv2.COLOR_BGR2RGB)

    # detect faces in the image
    faces = detector.detect_faces(RGBimageResized)
    for result in faces:
        plot_bounding_box(img, result, decimation, True, True)
    cv2.imshow('img', img)
    k = cv2.waitKey((30) & 0xff)
    if k == 27:
        break

print('MTCNN level 0 image size is :  ', width2, height2, color)
cap.release()
cv2.destroyAllWindows()







