# USAGE EXAMPLES
# for images:               python main.py --east frozen_east_text_detection.pb --image example_01.jpg
# for sequence translation: python main.py --translation-model checkpoint --sequence "How are you?"

import cv2
import tensorflow as tf

import arguments
import detection
import recognition
import translation

# do the tensor ops instantly
tf.enable_eager_execution()

if __name__ == "__main__":

    # parse the cli args
    arguments.parse_args()

    # image processing
    if arguments.ARGS.image:

        # load the input image
        image = cv2.imread(arguments.ARGS.image)

        # perform detection on the image and get the bounding boxes
        boxes = detection.perform(image)

        # perform recognition on the image with the help of the bounding boxes
        text = recognition.perform(image, boxes)

    # text processing
    else:
        print("wetf")
        # get seq argument value
        text = arguments.ARGS.sequence

    # perform translation on returned text and print to console
    print(translation.perform(text))
