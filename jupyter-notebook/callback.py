from enum import Enum
from typing import Callable, Any, Dict, Tuple

__all__ = ['Callback']


class Callback:
    """Class that represents callback functions.

    Has two variables: the function and its arguments. This class's call
    function is the preferred way of passing callback functions.
    """

    def __init__(self, func: Callable = None, *args, **kwargs):
        self._function: Callable = func
        self._args: Tuple = args
        self._kwargs: Dict[Any, Any] = kwargs

    def call(self, *args) -> Any:
        """Method for executing the callable.

        Args:
            *args: Any additional arguments that are passed to the callable.

        Returns:
            Any: The return of the callable.

        """
        self._function(*(self._args + args), **self._kwargs)
