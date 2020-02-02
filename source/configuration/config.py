import configparser
import os

from configuration.sections import SETTING_TO_SECTION
from helpers.consts import PATH_CONFIG

_PATH_DEFAULT_CONFIG = os.path.join(PATH_CONFIG, "default.ini")
_PATH_CURR_CONFIG = os.path.join(PATH_CONFIG, "config.ini")

_CONFIG = configparser.ConfigParser()
_CONFIG_DEFAULT = configparser.ConfigParser()


def load_config():
    global _CONFIG, _CONFIG_DEFAULT
    _CONFIG_DEFAULT.read(_PATH_DEFAULT_CONFIG)
    if not os.path.isfile(_PATH_CURR_CONFIG):
        _CONFIG.read(_PATH_DEFAULT_CONFIG)
        restore_defaults()
    else:
        _CONFIG.read(_PATH_CURR_CONFIG)


def restore_defaults():
    with open(_PATH_CURR_CONFIG, 'w') as config_file:
        global _CONFIG_DEFAULT
        _CONFIG_DEFAULT.write(config_file)


def _get_setting_section(setting):
    if not isinstance(setting, str):
        raise TypeError(f"Expected '{str}', got '{type(setting)}")
    if setting not in SETTING_TO_SECTION:
        raise ValueError(f"No such setting: '{setting}'")

    section = SETTING_TO_SECTION[setting]()
    return section


def get_setting_value(setting):
    global _CONFIG, _CONFIG_DEFAULT
    section = _get_setting_section(setting)
    invalid_config_section = section not in _CONFIG
    invalid_config_setting = False
    if not invalid_config_section:
        invalid_config_setting = setting not in _CONFIG[section]

    if invalid_config_setting or invalid_config_setting:
        with open(_PATH_CURR_CONFIG, 'w') as config_file:
            if invalid_config_section:
                _CONFIG[section] = _CONFIG_DEFAULT[section]
            elif invalid_config_setting:
                _CONFIG[section][setting] = _CONFIG_DEFAULT[section][setting]
            _CONFIG.write(config_file)

    return _CONFIG[section][setting]


def set_setting_value(setting, value):
    global _CONFIG
    section = _get_setting_section(setting)
    _CONFIG[section][setting] = value
