from . import definition
from .definition import Menu, ExitMenu


def start():
    # set image menu
    image_menu_options = [
        ("Set image", definition.set_image),
        ("Back", ExitMenu)
    ]
    image_menu = Menu(image_menu_options, description="[Image menu]")

    image_menu = Menu(image_menu_options, description="[Image menu]")

    main_menu_options = [
        (f"Set working image", image_menu),
        ("Other options", definition.other_options),
        ("Perform Detection", definition.detect),
        ("Perform Detection and Recognition", definition.detect_recongize),
        ("Perform Detection, Recognition and Translation", definition.detect_recognize_translate),
        ("Perform Recognition and Translation", definition.recognize_translate),
        ("Perform Recognition", definition.recognize),
        ("Perform Translation", definition.translate),
        ("Exit application", definition.exit_app)
    ]

    main_menu = Menu(main_menu_options, description="[Main menu]")
    main_menu.open()
