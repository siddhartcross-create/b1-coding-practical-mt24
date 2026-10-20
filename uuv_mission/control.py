"""Discrete-time PD controller for the UUV depth-tracking task.

This module is deliberately independent of ``dynamic.py``: the controller
takes plain floats in and returns a float, so it can be unit-tested on its
own and plugged into ``ClosedLoop`` without either side importing the other.
"""


class PDController:
    """Proportional-derivative controller.

    Control law (discrete time, sample period dt = 1):

        e[t] = r[t] - y[t]
        u[t] = KP * e[t] + KD * (e[t] - e[t-1]),   with e[-1] = 0

    where r is the reference depth, y the observed depth and u the action.
    The previous error e[t-1] is stored between calls, which is why this is
    a class rather than a plain function.
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        """Store the gains and start with no previous error (e[-1] = 0)."""
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0

    def reset(self) -> None:
        """Forget the stored error so a new simulation starts from e[-1] = 0."""
        self.prev_error = 0.0

    def act(self, reference: float, observation: float) -> float:
        """Return the control action u[t] for this step and remember e[t]."""
        error = reference - observation
        action = self.kp * error + self.kd * (error - self.prev_error)
        self.prev_error = error
        return action
