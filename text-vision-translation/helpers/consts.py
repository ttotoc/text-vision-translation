import os

# Paths
PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH_SRC_ROOT = os.path.dirname(os.path.dirname(__file__))
PATH_IMAGES = os.path.join(PATH_PROJECT_ROOT, 'images')
PATH_MODELS = os.path.join(PATH_PROJECT_ROOT, 'models')
PATH_MODEL_DETECTION = os.path.join(PATH_MODELS, "detection")
PATH_MODEL_TRANSLATION = os.path.join(PATH_MODELS, "translation")
PATH_CONFIG = os.path.join(PATH_PROJECT_ROOT, 'cfg')

EAST_RESIZE_MULT = 32
