import os

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

from configuration.config import get_setting_value
from configuration.settings import WIDTH_DETECTION, HEIGHT_DETECTION, EAST_TEXT_DETECTION, CONFIDENCE_DETECTION, \
    PADDING_DETECTION, IMAGE
from helpers.consts import PATH_MODEL_DETECTION


def perform(image):
    config_width = int(get_setting_value(WIDTH_DETECTION))
    config_height = int(get_setting_value(HEIGHT_DETECTION))
    config_padding = float(get_setting_value(PADDING_DETECTION))
    config_confidence = float(get_setting_value(CONFIDENCE_DETECTION))
    config_image = get_setting_value(IMAGE)

    # resize the image and grab the new image dimensions
    orig_image = image.copy()
    image = cv2.resize(image, (config_width, config_height))
    height, width = image.shape[:2]

    # define the two output layer names for the EAST detector model that we are interested in
    # the first is the output probabilities and the second can be used to derive the bounding box coordinates of text
    out_layers = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    model_path = os.path.join(PATH_MODEL_DETECTION, get_setting_value(EAST_TEXT_DETECTION))
    net = cv2.dnn.readNet(model_path)

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # make a forward prop and obtain the scores + geometrical data of the bounding boxes
    net.setInput(blob)
    scores, geometry = net.forward(out_layers)

    # use the scores array to grab the number of rows and columns
    rows, columns = scores.shape[2:4]
    # initialize our set of bounding box rectangles and their corresponding confidence scores
    rectangles = []
    confidences = []

    # loop over the number of rows
    for y in range(0, rows):
        # extract the scores (probabilities)
        scores_row = scores[0, 0, y]

        # extract geometrical data used to derive potential bounding box coordinates that surround text
        data_0 = geometry[0, 0, y]
        data_1 = geometry[0, 1, y]
        data_2 = geometry[0, 2, y]
        data_3 = geometry[0, 3, y]
        angles_row = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, columns):
            # ignore scores that don't have sufficient probability
            if scores_row[x] < config_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            offset_x, offset_y = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = angles_row[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = data_0[x] + data_2[x]
            w = data_1[x] + data_3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            end_x = int(offset_x + (cos * data_1[x]) + (sin * data_2[x]))
            end_y = int(offset_y - (sin * data_1[x]) + (cos * data_2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rectangles.append((start_x, start_y, end_x, end_y))

            confidences.append(scores_row[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rectangles), probs=confidences, overlapThresh=0.1)

    # bring the bounding boxes' ratio back to the original image's ratio

    orig_height, orig_width = orig_image.shape[:2]
    # determine the ratio change for both the width and height
    ratio_width = orig_width / float(config_width)
    ratio_height = orig_height / float(config_height)

    for i, (start_x, start_y, end_x, end_y) in enumerate(boxes):
        # scale the bounding box coordinates based on the respective ratios
        start_x = int(start_x * ratio_width)
        start_y = int(start_y * ratio_height)
        end_x = int(end_x * ratio_width)
        end_y = int(end_y * ratio_height)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        delta_x = int((end_x - start_x) * config_padding)
        delta_y = int((end_y - start_y) * config_padding)

        # apply padding to each side of the bounding box, respectively
        start_x = max(0, start_x - delta_x)
        start_y = max(0, start_y - delta_y)
        end_x = min(orig_width, end_x + (delta_x * 2))
        end_y = min(orig_height, end_y + (delta_y * 2))

        # update the coords
        boxes[i, 0] = start_x
        boxes[i, 1] = start_y
        boxes[i, 2] = end_x
        boxes[i, 3] = end_y

    # make a copy the output image
    output = orig_image.copy()

    # loop over the results and show them on the image
    for (start_x, start_y, end_x, end_y) in boxes:
        # draw the text and a bounding box surrounding the text region of the input image
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 0, 255), thickness=2)

    cv2.imshow(config_image, output)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return boxes
