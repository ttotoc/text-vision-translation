import configparser
import os

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


def get_setting_value(setting):
    global _CONFIG, _CONFIG_DEFAULT
    invalid_config_section = setting.section not in _CONFIG
    invalid_config_setting = False
    if not invalid_config_section:
        invalid_config_setting = setting not in _CONFIG[setting.section]

    if invalid_config_setting or invalid_config_setting:
        with open(_PATH_CURR_CONFIG, 'w') as config_file:
            if invalid_config_section:
                _CONFIG[setting.section] = _CONFIG_DEFAULT[setting.section]
            elif invalid_config_setting:
                _CONFIG[setting.section][setting] = _CONFIG_DEFAULT[setting.section][setting]
            _CONFIG.write(config_file)

    return _CONFIG[setting.section][setting]


def set_setting_value(setting, value):
    if setting.set_value(value):
        global _CONFIG
        _CONFIG[setting.section][setting] = value
        with open(_PATH_CURR_CONFIG, 'w') as config_file:
            _CONFIG.write(config_file)
