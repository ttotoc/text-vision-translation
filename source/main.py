# USAGE EXAMPLES
# for images:               python main.py --east frozen_east_text_detection.pb --image example_01.jpg
# for sequence translation: python main.py --translation-model checkpoint --sequence "How are you?"

from os import environ
import sys
import cv2
import tensorflow as tf

import arguments
import detection
import recognition
from translation import translation
from menu import menu_setup

# show only error logs
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# do the tensor ops instantly
tf.enable_eager_execution()


if __name__ == "__main__":

    # if no arguments were provided at cli, just start the main menu and don't bother to parse anything
    if len(sys.argv) > 1:

        # parse the cli args
        arguments.parse_args()

        if arguments.ARGS.image:

            # load the input image
            image = cv2.imread(arguments.ARGS.image)

            # perform detection on the image and get the bounding boxes, if east_model was provided
            if arguments.ARGS.east_model:
                boxes = detection.perform(image)

                if arguments.ARGS.recognition:
                    # perform recognition on the image with the help of the bounding boxes
                    text = recognition.perform(image, boxes)
            else:
                # perform recognition on the image without bounding boxes
                text = recognition.perform(image)

        # text processing
        else:

            # get seq argument value
            text = [arguments.ARGS.sequence]

        # perform translation on returned text and print to console
        if arguments.ARGS.translation_model and arguments.ARGS.recognition:
            translation.perform(text)

    #print(arguments.ARGS)
    menu_setup.start()