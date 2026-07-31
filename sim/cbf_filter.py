"""Phase 1 CBF safety filter: single-integrator, single circular keep-out zone.

Model:    p_dot = u,           p, u in R^2   (body (x,y), velocity command)
Barrier:  h(p)  = ||p - p_o||^2 - r^2        (h >= 0  <=>  safe)
Filter:   u* = argmin ||u - u_des||^2  s.t.  h_dot + alpha(h) >= 0

With a single constraint the QP is a projection onto a half-space, so there is a
closed form -- no QP solver needed. Multiple obstacles stack constraints and
require a real QP (that is Phase 4).

The RL locomotion policy is treated as the low-level velocity tracker: we only
shape the velocity *command* it receives, so the body is a kinematic point here.
"""

import numpy as np


class CircularKeepOut:
    """A virtual circular forbidden region in world coordinates."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)  # p_o, shape (2,)
        self.radius = float(radius)                    # r

    def h(self, p):
        """Barrier value. >= 0 is safe, 0 is on the boundary, < 0 is inside."""
        p = np.asarray(p, dtype=float)
        return float((p - self.center) @ (p - self.center) - self.radius**2)

    def grad_h(self, p):
        """dh/dp = 2 (p - p_o)."""
        p = np.asarray(p, dtype=float)
        return 2.0 * (p - self.center)


class SingleIntegratorCBF:
    """Closed-form CBF safety filter for one circular keep-out zone.

    alpha is the extended class-K function in the CBF condition. We use the
    linear form alpha(h) = gamma * h, the standard first choice; larger gamma
    lets the filter act later/more aggressively, smaller gamma is more cautious.
    """

    def __init__(self, obstacle: CircularKeepOut, gamma: float = 1.0):
        self.obstacle = obstacle
        self.gamma = float(gamma)

    def alpha(self, h):
        return self.gamma * h

    def filter(self, p, u_des):
        """Return (u_safe, info).

        Solves   min ||u - u_des||^2   s.t.   a . u >= b
        with     a = grad_h(p),   b = -alpha(h(p)).

        Closed form: if u_des already satisfies the constraint, return it
        unchanged; otherwise project it onto the constraint boundary.
        """
        p = np.asarray(p, dtype=float)
        u_des = np.asarray(u_des, dtype=float)

        h = self.obstacle.h(p)
        a = self.obstacle.grad_h(p)          # constraint normal
        b = -self.alpha(h)                   # required: a . u >= b

        aa = float(a @ a)
        slack = float(a @ u_des) - b         # >= 0 means u_des is already safe

        if aa < 1e-12:
            # Degenerate: exactly at the obstacle center, gradient vanishes.
            # Nothing the filter can do here; return desired and flag it.
            return u_des.copy(), {"h": h, "active": False, "degenerate": True,
                                  "slack": slack}

        if slack >= 0.0:
            return u_des.copy(), {"h": h, "active": False, "degenerate": False,
                                  "slack": slack}

        # Project onto the half-space boundary a . u = b.
        u_safe = u_des + (-slack / aa) * a
        return u_safe, {"h": h, "active": True, "degenerate": False,
                        "slack": slack}
