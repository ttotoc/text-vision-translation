from imutils.object_detection import non_max_suppression
import numpy as np
import cv2


def perform(image):

    # get cli arguments
    from arguments import ARGS

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (ARGS.width, ARGS.height))
    height, width = image.shape[:2]

    # define the two output layer names for the EAST detector model that we are interested in
    # the first is the output probabilities and the second can be used to derive the bounding box coordinates of text
    out_layers = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(ARGS.east_model)

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # make a forward prop and obtain the scores + geometrical data of the bounding boxes
    net.setInput(blob)
    scores, geometry = net.forward(out_layers)

    # use the scores array to grab the number of rows and columns
    rows, columns = scores.shape[2:4]
    # initialize our set of bounding box rectangles and corresponding confidence scores
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
            if scores_row[x] < ARGS.min_confidence_roi:
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
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)
    return boxes
