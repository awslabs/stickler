"""Time utility functions using threading.Event for sleep operations."""

from threading import Event


def sleep(timeout: float) -> None:
    """
    Sleep for the specified number of seconds using threading.Event.

    This provides a non-blocking sleep that can be interrupted if needed,
    unlike time.sleep() which blocks the thread entirely.

    Args:
        timeout: Number of seconds to sleep (can be fractional)
    """
    event = Event()
    event.wait(timeout=timeout)
