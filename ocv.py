from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import tensorflow as tf
import pytesseract
import matplotlib as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help='path to input image')
ap.add_argument("-east", "--east", type=str, help='path to input EAST text detector')
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width")
ap.add_argument("-e", "--height", type=int, default=320, help='resized image height')
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

print("[INFO] loading EAST text detector,,,")
net = cv2.dnn.readNet(args["east"])

# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123, 68, 116.78, 103.94), swapRB=True, crop=False)
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
cv2.imshow("a", image)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()
print("[INFO] text detection took {:.6f} seconds".format(end - start))
# print("scores:", scores)
# print("geometry:", geometry)
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    # extract the scores (probabilities), followed by the geometrical
    # data used to derive potential bounding box coordinates that
    # surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    # loop over the number of columns
    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < args["min_confidence"]:
            continue

        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive the width and height of
        # the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to
        # our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
#
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image

results = []

for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    r = orig[startX:endY, startX:endX]
    configuration = ("-l eng --oem 1 --psm 8")
    text = pytesseract.image_to_string(r, config=configuration)
    results.append(((startX, startY, endX, endY), text))
# Display the image with bounding box and recognized text
orig_image = orig.copy()

# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
    # display the text detected by Tesseract
    print("{}\n".format(text))

    # Displaying text
    text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
    cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                  (0, 0, 255), 2)
    cv2.putText(orig_image, text, (start_X, start_Y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

print(results)

cv2.imshow("Text Detection", orig_image)
cv2.waitKey(0)
