"""Unit tests for uuv_mission.control.PDController."""
import pytest

from uuv_mission.control import PDController


def test_first_call_uses_zero_previous_error():
    # e[0] = 1 - 0 = 1, e[-1] = 0  ->  u = 0.15*1 + 0.6*(1 - 0) = 0.75
    controller = PDController()
    assert controller.act(reference=1.0, observation=0.0) == pytest.approx(0.75)


def test_second_call_uses_stored_previous_error():
    # e[0] = 1.0, e[1] = 0.5  ->  u = 0.15*0.5 + 0.6*(0.5 - 1.0) = -0.225
    controller = PDController()
    controller.act(reference=1.0, observation=0.0)
    assert controller.act(reference=1.0, observation=0.5) == pytest.approx(-0.225)


def test_reset_clears_previous_error():
    controller = PDController()
    controller.act(reference=1.0, observation=0.0)
    controller.reset()
    # After reset the call must behave exactly like a first call again.
    assert controller.prev_error == 0.0
    assert controller.act(reference=1.0, observation=0.0) == pytest.approx(0.75)


def test_custom_gains_are_used():
    # e[0] = 2.0, e[-1] = 0  ->  u = 1.0*2.0 + 0.5*(2.0 - 0) = 3.0
    controller = PDController(kp=1.0, kd=0.5)
    assert controller.act(reference=2.0, observation=0.0) == pytest.approx(3.0)
