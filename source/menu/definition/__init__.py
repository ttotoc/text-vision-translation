from os.path import join as path_join

import cv2

import arguments
import detection
import recognition
from arguments import ARGS
from translation import translation


class Menu:
    HELP_OPTION = "help"
    MENU_CHANGED = True

    def __init__(self, option_args, description=None):
        '''
        :param option_args: a list of (text, handler_or_menu) tuples that define the options
        '''

        self.description = description

        for opt in option_args:
            if not isinstance(opt, tuple):
                raise TypeError(f'Expected {tuple}, got {type(opt)}')
            if not isinstance(opt[0], str):
                raise TypeError(f'Expected {str}, got {type(opt[0])}')
            if not callable(opt[1]) and not isinstance(opt[1], Menu) and not opt[1] is ExitMenu:
                raise TypeError(f'Expected {callable}, {Menu} or {ExitMenu}, got {type(opt[1])}')

        self.options = option_args

    def open(self):
        '''
        :param input_val: an index of the option list
        '''
        self.help()

        while True:

            input_val = input('> ')

            # check for other text options
            if isinstance(input_val, str) and input_val == Menu.HELP_OPTION:
                self.help()
                continue

            # validate input
            try:
                input_val = int(input_val)
            except Exception as ex:
                print("Invalid option")
                continue

            if not 1 <= input_val <= len(self.options):
                print(f'Invalid option(must be between {1} and {len(self.options)})')
                continue

            # execute option function, open menu or exit menu
            option_idx = input_val - 1
            option_action = self.options[option_idx][1]
            if option_action is ExitMenu:
                break
            elif isinstance(option_action, Menu):
                option_action.open()
            else:
                option_action()

            self.help()

    def help(self):
        '''
        prints the options and description of menu
        '''
        print(self.description)
        for i, opt in enumerate(self.options):
            print(f'{i + 1}. {opt[0]}')


# used when you want for an option to exit from the current menu
class ExitMenu:
    pass


# FUNCTIONS FOR OPTIONS

# set image option
def set_image():
    input_name = input("Enter the file name: ")
    img_path = path_join(arguments.IMAGES_DIR, input_name)
    if arguments.image_exists(img_path):
        ARGS.image = img_path
        print(f"Working image changed to: {input_name}")
    else:
        print(f"Invalid image name: {input_name}")


# detection
def detect():
    image = cv2.imread(arguments.ARGS.image)
    detection.perform(image)


# detection & recognition
def detect_recongize():
    image = cv2.imread(arguments.ARGS.image)
    boxes = detection.perform(image)
    recognition.perform(image, boxes)


def detect_recognize_translate():
    image = cv2.imread(arguments.ARGS.image)
    boxes = detection.perform(image)
    text = recognition.perform(image, boxes)
    translation.perform(text)


def recognize_translate():
    image = cv2.imread(arguments.ARGS.image)
    text = recognition.perform(image)
    print(f"Recognized text: {text}")
    translation.perform(text)


def recognize():
    image = cv2.imread(arguments.ARGS.image)
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
