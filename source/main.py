# USAGE
# python main.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python main.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

import cv2

import arguments
import detection
import recognition

if __name__ == "__main__":

    # parse the cli args
    arguments.parse_args()

    # load the input image
    image = cv2.imread(arguments.ARGS.image)

    # perform detection on the image and get the bounding boxes
    boxes = detection.perform(image)

    # perform recognition on the image with the help of the bounding boxes
    recognition.perform(image, boxes)
