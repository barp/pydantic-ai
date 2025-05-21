import uuid
from typing import Any


def get_callable_name(callable_: Any) -> str:
    # TODO: Need to improve this...
    return getattr(callable_, '__name__', str(callable_))


def get_unique_string() -> str:
    # TODO: Need to improve this...
    return str(uuid.uuid4())
