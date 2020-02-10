import os
import sys

import cv2

import detection
import recognition
from configuration.config import get_setting_value, set_setting_value
from configuration.settings import IMAGE
from helpers.consts import PATH_IMAGES
from menu.menu import Menu
from translation import translation


def start():
    # image submenu
    image_menu_options = [
        ("Set image", set_image),
        ("Back", None)
    ]
    image_menu = Menu(image_menu_options, description="[Image Menu]")

    # settings menu
    settings_menu_options = [
        ("EAST Model", None),
        ("Translation Model", None),
        ("Resize Width", None),
        ("Resize Height", None),
        ("Padding", None),
        ("Minimum Confidence", None),
        ("Tesseract Parameters", None),
        ("Concatenate Recognition Results", None),
        # ("Concatenate Recognition Results", ExitMenu)
    ]
    settings_menu = Menu(settings_menu_options, description="[Settings Menu]")

    # main menu
    main_menu_options = [
        ("Set working image", image_menu),
        ("Settings", settings_menu),
        ("Perform Detection", detect),
        ("Perform Detection and Recognition", detect_recognize),
        ("Perform Detection, Recognition and Translation", detect_recognize_translate),
        ("Perform Recognition and Translation", recognize_translate),
        ("Perform Recognition", recognize),
        ("Perform Translation", translate),
        ("Exit application", exit_app)
    ]
    main_menu = Menu(main_menu_options, description="[Main Menu]")

    main_menu.open()


# set image option
def set_image():
    input_name = input("Enter the file name: ")
    set_setting_value(IMAGE, input_name)
    print(f"Working image changed to: {input_name}")


# detection
def detect():
    img_path = os.path.join(PATH_IMAGES, get_setting_value(IMAGE))
    image = cv2.imread(img_path)
    print(get_setting_value(IMAGE))
    detection.perform(image)


# detection & recognition
def detect_recognize():
    img_path = os.path.join(PATH_IMAGES, get_setting_value(IMAGE))
    image = cv2.imread(img_path)
    boxes = detection.perform(image)
    recognition.perform(image, boxes)


def detect_recognize_translate():
    img_path = os.path.join(PATH_IMAGES, get_setting_value(IMAGE))
    image = cv2.imread(img_path)
    boxes = detection.perform(image)
    text = recognition.perform(image, boxes)
    translation.perform(text)


def recognize_translate():
    img_path = os.path.join(PATH_IMAGES, get_setting_value(IMAGE))
    image = cv2.imread(img_path)
    text = recognition.perform(image)
    print(f"Recognized text: {text}")
    translation.perform(text)


def recognize():
    img_path = os.path.join(PATH_IMAGES, get_setting_value(IMAGE))
    image = cv2.imread(img_path)
    text = recognition.perform(image)
    print(f"Recognized text: {text}")


def translate():
    text = [input("Enter text to translate: ")]
    translation.perform(text)
    input("Press Enter to continue...")


def exit_app():
    sys.exit()
