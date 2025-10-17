import pytest

from lambdapic.core.utils.enable_mixin import EnableMixin, enabled_method, if_enabled


class DummyController(EnableMixin):
    def __init__(self) -> None:
        self.events: list = []

    @enabled_method
    def record(self, value: int) -> int:
        """Append a value when enabled and return current event count."""
        self.events.append(value)
        return len(self.events)

    @if_enabled
    def flag(self, value: str) -> bool:
        """Append a flag entry when enabled."""
        self.events.append(("flag", value))
        return True


def test_enabled_method_behaves_as_descriptor():
    with pytest.raises(AttributeError):
        DummyController.record


def test_enable_mixin_decorators_respect_state():
    controller = DummyController()
    descriptor = DummyController.__dict__["record"]

    # default state is enabled
    assert controller.is_enabled()
    assert controller.record.__name__ == "record"
    assert controller.record.__doc__ == descriptor.func.__doc__

    assert controller.record(1) == 1
    assert controller.flag("start") is True
    assert controller.events == [1, ("flag", "start")]

    controller.disable()
    assert controller.record(2) is None
    assert controller.flag("stop") is None
    assert controller.events == [1, ("flag", "start")]

    controller.enable()
    record_count = controller.record(3)
    assert record_count == len(controller.events)
    assert controller.events[-1] == 3
    assert controller.flag("resume") is True
    assert controller.events == [1, ("flag", "start"), 3, ("flag", "resume")]
