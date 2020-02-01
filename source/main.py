from os import environ
import sys
import cv2
import tensorflow as tf

import detection
import recognition
from translation import translation
from menu.main_menu import start as main_menu_start

# show only error logs
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# do the tensor ops instantly
tf.enable_eager_execution()


if __name__ == "__main__":
    main_menu_start()