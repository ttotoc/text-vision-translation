from configuration.settings import *

_INPUT_SECTION = "Input"
_MODELS_SECTION = "Models"
_DETECTION_SECTION = "Detection"
_RECOGNITION_SECTION = "Recognition"
_TRANSLATION_SECTION = "Translation"


def input():
    return _INPUT_SECTION


def models():
    return _MODELS_SECTION


def detection():
    return _DETECTION_SECTION


def recognition():
    return _RECOGNITION_SECTION


def translation():
    return _TRANSLATION_SECTION


SETTING_TO_SECTION = {
    IMAGE: input,
    EAST_TEXT_DETECTION: models,
    TRANSLATION: models,
    WIDTH_DETECTION: detection,
    HEIGHT_DETECTION: detection,
    PADDING_DETECTION: detection,
    CONFIDENCE_DETECTION: detection,
    TESSERACT_PARAMS: recognition,
    CONCATENATE: recognition,
    TEXT_SEQUENCE: translation,
}
