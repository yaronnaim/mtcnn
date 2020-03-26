import mtcnn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle, Circle
import cv2
import numpy as np


# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for face in result_list:
		# get coordinates
		x, y, width, height = face['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for key, value in face['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)

	# show the plot
	pyplot.show()


# draw each face separately
def draw_faces(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(data[y1:y2, x1:x2])
	# show the plot
	pyplot.show()


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
	decimation = 8
	height2 = np.int(height/decimation)
	width2 = np.int(width/decimation)
	img = cv2.resize(img, (height2, width2))

	RGBimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# detect faces in the image
	faces = detector.detect_faces(RGBimage)
	for result in faces:
		x, y, w, h = result['box']
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
		roi_gray = RGBimage[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		for key, value in result['keypoints'].items():
			cv2.circle(img, value, radius=2, color=(255,0,0), thickness=2)
	cv2.imshow('img', img)
	k = cv2.waitKey((30) & 0xff)
	if k == 27:
		break

print('Frame size is ', width2, height2, color)
cap.release()
cv2.destroyAllWindows()




