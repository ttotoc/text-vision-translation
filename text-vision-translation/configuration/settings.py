import os

from configuration.sections import *
from configuration.setting import Setting
from helpers.consts import PATH_IMAGES, PATH_MODEL_DETECTION, PATH_MODEL_TRANSLATION, PICKLE_EXTENSION, EAST_RESIZE_MULT


def _validate_image_path(img_name):
    img_path = os.path.join(PATH_IMAGES, img_name)
    if not os.path.isfile(img_path):
        return f"File not found: {img_name}"


def _validate_east_path(east_name):
    east_path = os.path.join(PATH_MODEL_DETECTION, east_name)
    if not os.path.isfile(east_path):
        return f"EAST Model not found: {east_name}"


def _validate_translation_mdl_path(translation_mdl_name):
    translation_mdl_path = os.path.join(PATH_MODEL_TRANSLATION, translation_mdl_name)
    if not os.path.isfile(translation_mdl_path):
        return f"Translation Model not found: {translation_mdl_name}"
    translation_params_path = os.path.join(PATH_MODEL_TRANSLATION, PICKLE_EXTENSION)
    if not os.path.isfile(translation_params_path):
        return f"The params({PICKLE_EXTENSION} file) associated with the model could not be found"


def _validate_width_detection(width):
    if width % EAST_RESIZE_MULT != 0:
        return f"Resized width must be a multiple of 32."


def _validate_height_detection(height):
    if height % EAST_RESIZE_MULT != 0:
        return f"Resized height must be a multiple of 32."


def _validate_padding_detection(padding):
    if padding < 0:
        return "Padding cannot be negative."


def _validate_confidence_detection(confidence):
    if not 0 <= confidence <= 1:
        return "Confidence must be between 0-1."


IMAGE = Setting("Image", INPUT_SECTION, validator=_validate_image_path)
EAST_TEXT_DETECTION = Setting("EastTextDetection", MODELS_SECTION, validator=_validate_east_path)
TRANSLATION = Setting("Translation", MODELS_SECTION, validator=_validate_translation_mdl_path)
WIDTH_DETECTION = Setting("Width", DETECTION_SECTION, int, _validate_width_detection)
HEIGHT_DETECTION = Setting("Height", DETECTION_SECTION, int, _validate_height_detection)
PADDING_DETECTION = Setting("Padding", DETECTION_SECTION, float, _validate_padding_detection)
CONFIDENCE_DETECTION = Setting("Confidence", DETECTION_SECTION, float, _validate_confidence_detection)
TESSERACT_PARAMS = Setting("TesseractParams", RECOGNITION_SECTION)
CONCATENATE = Setting("Concatenate", RECOGNITION_SECTION, bool)
TEXT_SEQUENCE = Setting("Sequence", TRANSLATION_SECTION)
