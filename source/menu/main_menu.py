import os

import cv2

from helpers.consts import PATH_IMAGES
from .menu import Menu, ExitMenu
from configuration.config import get_setting_value, set_setting_value
from configuration.settings import IMAGE


def start():
    # image submenu
    image_menu_options = [
        ("Set image", set_image),
        ("Back", ExitMenu)
    ]
    image_menu = Menu(image_menu_options, description="[Image menu]")

    # main menu
    main_menu_options = [
        ("Set working image", image_menu),
        ("Settings", other_options),
        ("Perform Detection", detect),
        ("Perform Detection and Recognition", detect_recognize),
        ("Perform Detection, Recognition and Translation", detect_recognize_translate),
        ("Perform Recognition and Translation", recognize_translate),
        ("Perform Recognition", recognize),
        ("Perform Translation", translate),
        ("Exit application", exit_app)
    ]
    main_menu = Menu(main_menu_options, description="[Main menu]")

    main_menu.open()


# set image option
def set_image():
    input_name = input("Enter the file name: ")
    img_path = os.path.join(PATH_IMAGES, input_name)
    if not os.path.isfile(img_path):
        set_setting_value(IMAGE, img_path)
        print(f"Working image changed to: {input_name}")
    else:
        print(f"Invalid image name: {input_name}")


# detection
def detect():
    image = cv2.imread(get_setting_value)
    detection.perform(image)


# detection & recognition
def detect_recognize():
    image = cv2.imread(IMAGE)
    boxes = detection.perform(image)
    recognition.perform(image, boxes)


def detect_recognize_translate():
    image = cv2.imread(IMAGE)
    boxes = detection.perform(image)
    text = recognition.perform(image, boxes)
    translation.perform(text)


def recognize_translate():
    image = cv2.imread(IMAGE)
    text = recognition.perform(image)
    print(f"Recognized text: {text}")
    translation.perform(text)


def recognize():
    image = cv2.imread(IMAGE)
    text = recognition.perform(image)
    print(f"Recognized text: {text}")


def translate():
    text = [input("Enter text to translate: ")]
    translation.perform(text)


def other_options():
    print("WIP")


def exit_app():
    import sys
    sys.exit()
