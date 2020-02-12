from types import FunctionType


class Setting:
    SUPPORTED_TYPES = frozenset([int, float, str, bool])

    def __init__(self, name, section, val_type=str, validator=None):
        if not isinstance(section, str):
            raise TypeError(f"Expected '{str}', got '{type(section)}")
        if val_type not in Setting.SUPPORTED_TYPES:
            raise TypeError(f'Setting type not supported: {type(val_type)}')
        if not isinstance(validator, FunctionType) and validator is not None:
            raise TypeError(f'Expected {FunctionType}, got {type(validator)}')
        self._name = name
        self._section = section
        self._type = val_type
        self._validator = validator
        self._value = None

    @property
    def value(self):
        return self._value

    def set_value(self, new_value):
        if self._validator:
            validate_err_msg = self._validator(new_value)
            if validate_err_msg:
                print(validate_err_msg)
                return False
        self._value = self._type(new_value)
        return self._value

    @property
    def name(self):
        return self._section

    @property
    def section(self):
        return self._section
