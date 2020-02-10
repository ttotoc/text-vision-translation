from types import FunctionType


class Menu:
    HELP_OPTION = "help"
    MIN_OPTION_IDX = 1

    def __init__(self, option_args, description=None):
        """
        :param option_args: a list of (text, handler_or_menu) tuples that define the options
        """

        self.description = description

        for opt in option_args:
            if not isinstance(opt, tuple):
                raise TypeError(f'Expected {tuple}, got {type(opt)}')
            if not isinstance(opt[0], str):
                raise TypeError(f'Expected {str}, got {type(opt[0])}')
            if not isinstance(opt[1], FunctionType) and not isinstance(opt[1], Menu) and not opt[1] is None:
                raise TypeError(f'Expected {FunctionType}, {Menu} or {None}, got {type(opt[1])}')

        self.options = option_args

    def open(self):
        self.help()

        while True:

            input_val = input('> ')

            # check for other text options
            if input_val == Menu.HELP_OPTION:
                self.help()
                continue

            # validate input
            if input_val.isnumeric():
                input_val = int(input_val)
            else:
                print("Invalid option")
                continue

            if not Menu.MIN_OPTION_IDX <= input_val <= len(self.options):
                print(f'Invalid option(must be between {1} and {len(self.options)})')
                continue

            # execute option function, open menu or exit menu
            option_idx = input_val - 1
            option_action = self.options[option_idx][1]
            if option_action is None:
                break
            elif isinstance(option_action, Menu):
                option_action.open()
            else:
                option_action()

            self.help()

    def help(self):
        """
        prints the options and description of menu
        """
        print(self.description)
        for i, opt in enumerate(self.options):
            print(f'{i + 1}. {opt[0]}')
