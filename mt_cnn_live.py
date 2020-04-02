import mtcnn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle, Circle
import cv2
import numpy as np


def plot_bounding_box(image, result, scale=1, facial_landmarks=False):
    """

    :param image:  Original size image
    :param result: MTCNN output, Dict keys 'box' and 'keypoints'
    :param scale:  Scale down factor betwwen original to MTCNN image
    :param facial_landmarks: Bool to indicate if facial landmark should be
    plotted
    :return: None
    """
    x, y, w, h = [i*scale for i in result['box']]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if facial_landmarks == False:
        return
    else:
        for key, value in result['keypoints'].items():
            cv2.circle(image, (value[0]*scale, value[1]*scale),
                                radius=2,
                                color=(255, 0, 0),
                                thickness=2)


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
    decimation = 1
    height2 = np.int(height/decimation)
    width2 = np.int(width/decimation)

    # Resize and convert to RGB
    RGBimageResized = cv2.cvtColor(cv2.resize(img, (height2, width2)),
                                  cv2.COLOR_BGR2RGB)

    # detect faces in the image
    faces = detector.detect_faces(RGBimageResized)
    for result in faces:
        plot_bounding_box(img, result, decimation, True)
    cv2.imshow('img', img)
    k = cv2.waitKey((30) & 0xff)
    if k == 27:
        break

print('MTCNN level 0 image size is :  ', width2, height2, color)
cap.release()
cv2.destroyAllWindows()







