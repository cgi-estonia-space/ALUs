import asyncio
from typing import Any


class Timer:
    def __init__(self, timeout, callback):
        self._task: asyncio.Task = None
        self._timeout: int = timeout
        self._callback: Any = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()


def debounce(wait):
    def decorator(fn):
        timer: Timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator
