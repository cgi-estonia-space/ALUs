import abc
from enum import Enum


class OptionsInterface(Enum, metaclass=Enum.__class__):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'get_normal_name') \
               and callable(subclass.get_normal_name) or NotImplementedError

    @abc.abstractmethod
    def get_normal_name(self) -> str:
        """Display human-readable name of the option"""
        raise NotImplementedError
