from functools import wraps


class EnableMixin:
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        if not hasattr(self, '_enabled'):
            self._enabled = True
        return self._enabled
    
class enabled_method:
    def __init__(self, func):
        self.func = func
    
    def __get__(self, instance, owner):
        if instance is None:
            raise AttributeError("Cannot call method on class without instance")
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            if instance.is_enabled():
                return self.func(instance, *args, **kwargs)
            else:
                return
        return wrapper
    
def if_enabled(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_enabled():
            return func(self, *args, **kwargs)
        else:
            return
    return wrapper
