from __future__ import annotations

import asyncio
import functools
import time
from abc import ABCMeta


class CallbackMeta(ABCMeta):
    """
    A metaclass that ensures subclasses receive the same callback decoration as their base class.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and not getattr(
                attr_value, "_with_callbacks", False
            ):
                # Wrap subclass methods if the base class method is wrapped
                if any(
                    getattr(getattr(base, attr_name, None), "_with_callbacks", False)
                    for base in bases
                ):
                    namespace[attr_name] = with_callbacks(attr_value)
        return super().__new__(mcls, name, bases, namespace, **kwargs)


def with_callbacks(method):
    """
    Decorator that calls _pre\\_*method* and _post\\_*method* callbacks
    if they exist, passing the result of *method* to the post callback.
    """
    method_name = method.__name__
    pre_cb = f"_pre_{method_name}"
    post_cb = f"_post_{method_name}"
    setattr(method, "_with_callbacks", True)

    if asyncio.iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_wrapper(self, *args, **kwargs):
            if hasattr(self, pre_cb):
                getattr(self, pre_cb)(*args, **kwargs)
            start_time = time.time()
            result = await method(self, *args, **kwargs)
            duration = time.time() - start_time
            if hasattr(self, post_cb):
                new_result = getattr(self, post_cb)(result, duration, *args, **kwargs)
                if new_result is not None:
                    result = new_result
            return result

        return async_wrapper
    else:

        @functools.wraps(method)
        def sync_wrapper(self, *args, **kwargs):
            if hasattr(self, pre_cb):
                getattr(self, pre_cb)(*args, **kwargs)
            start_time = time.time()
            result = method(self, *args, **kwargs)
            duration = time.time() - start_time
            if hasattr(self, post_cb):
                new_result = getattr(self, post_cb)(result, duration, *args, **kwargs)
                if new_result is not None:
                    result = new_result
            return result

        return sync_wrapper
