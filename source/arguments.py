from os.path import join as path_join
from os.path import isfile

ARGS = None

TEXT_DETECTION_MODELS_DIR = "../models/text_detection/"
TRANSLATION_MODELS_DIR = "../models/translation/"
IMAGES_DIR = "../images/"


def parse_args():
    # CLI Arguments library
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-img",
        "--image",
        type=str,
        help="input image(located in ../images/). NOTE: This cannot be used with -s at the same time"
    )
    ap.add_argument(
        "-east",
        "--east-model",
        type=str,
        default=path_join(TEXT_DETECTION_MODELS_DIR, "frozen_east_text_detection.pb"),
        help="EAST text detector model(with the extension, located in ../models/text_detection/)"
    )
    ap.add_argument(
        "-tm",
        "--translation-model",
        type=str,
        required=True,
        help="name of the translation model(without extension, located in ../models/translation/)"
    )
    ap.add_argument(
        "-s",
        "--sequence",
        type=str,
        help="the sequence of words to be translated. NOTE: This cannot be used with -img at the same time"
    )
    ap.add_argument(
        "-c",
        "--min-confidence-roi",
        type=float,
        default=0.5,
        help="minimum probability required to inspect a possible region of interest"
    )
    ap.add_argument(
        "-w",
        "--width",
        type=int,
        default=320,
        help="nearest multiple of 32 for resized width"
    )
    ap.add_argument(
        "-e",
        "--height",
        type=int,
        default=320,
        help="nearest multiple of 32 for resized height"
    )
    ap.add_argument(
        "-p",
        "--padding",
        type=float,
        default=0.0,
        help="amount of padding to add to each border of a region of interest(0.0 - 1.0)"
    )

    # set the arguments global var
    global ARGS
    ARGS = ap.parse_args()

    # image OR sequence of words as input, not both simultaneously
    if ARGS.image and ARGS.sequence:
        ap.error("--image and --sequence parameters cannot be used in the same run instance.")
    if ARGS.image:
        ARGS.image = path_join(IMAGES_DIR, ARGS.image)

        if not isfile(ARGS.image):
            ap.error("Input image could not be found.")

        # east model
        ARGS.east_model = path_join(TEXT_DETECTION_MODELS_DIR, ARGS.east_model)
        if not isfile(ARGS.east_model):
            ap.error("Detector model could not be found.")

        if ARGS.width % 32 != 0:
            ap.error("Resized width must be a multiple of 32.")

        if ARGS.height % 32 != 0:
            ap.error("Resized height must be a multiple of 32.")

        if not 0.0 <= ARGS.padding:
            ap.error("Padding cannot be negative.")
    elif not ARGS.sequence:
        ap.error("Neither the --image nor the --sequence parameter have been introduced.")

    # translation model
    model_path = path_join(TRANSLATION_MODELS_DIR, ARGS.translation_model)
    params_path = path_join(TRANSLATION_MODELS_DIR, ARGS.translation_model) + ".pickle"
    if not isfile(model_path + ".index"):
        raise FileNotFoundError("The .index file associated with the model could not be found.")
    if not isfile(params_path):
        raise FileNotFoundError("The params(.pickle file) associated with the model could not be found.")
    # modify the argument to store a dict of model and params paths
    ARGS.translation_model = {
        "model": model_path,
        "params": params_path
    }
