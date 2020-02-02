from os import environ
from configuration.config import load_config
from menu.main_menu import start as main_menu_start

# show only error logs
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == "__main__":
    load_config()
    main_menu_start()