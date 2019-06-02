from os.path import join as path_join

ARGS = None

MODELS_DIR = "../models/"
IMAGES_DIR = "../images/"


def parse_args():

    # CLI Arguments library
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-img",
        "--image",
        required=True,
        type=str,
        help="path to input image"
    )
    ap.add_argument(
        "-east",
        "--east-model",
        type=str,
        default=path_join(MODELS_DIR, "frozen_east_text_detection.pb"),
        help="path to input EAST text detector model"
    )
    ap.add_argument(
        "-c",
        "--min-confidence",
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

    # set the arguments
    global ARGS
    ARGS = ap.parse_args()

    # args validation
    ARGS.image = path_join(IMAGES_DIR, ARGS.image)

    ARGS.east_model = path_join(MODELS_DIR, ARGS.east_model)

    if ARGS.width % 32 != 0:
        raise ValueError("Resized width must be a multiple of 32.")

    if ARGS.height % 32 != 0:
        raise ValueError("Resized height must be a multiple of 32.")

    if not 0.0 <= ARGS.padding:
        raise ValueError("Padding cannot be negative.")