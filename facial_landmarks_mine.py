import numpy as np
import argparse
import cv2
import dlib
import imutils


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((62, 2), dtype=dtype)

    for i in range(62):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
print("Detector: ", detector)
print("Predictor: ", predictor)

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
print("Rects: ", rects)
print("Rect type: ", type(rects[0]))
print("Rects length: ", len(rects))


for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    print("Shape: ", shape)
    shape = shape_to_np(shape)

    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(i+1), (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)
