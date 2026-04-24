"""Agent-facing helpers that every PolySignalLab component should integrate.

This subpackage is the component-side counterpart of ``PSLAgent``. Anything a
supervised component needs to cooperate with the per-host agent — enrollment,
one-shot 'I'm alive' announce for adoption by the agent, future graceful-
shutdown hooks, future healthcheck responders — lives here so components have
exactly one import path for all of it:

    from shared.pslagent import announce_alive, enroll_if_requested

Components should call these helpers at the top of ``main()`` before any
other startup work. Each helper is a no-op when its corresponding env var /
key / signal is absent, so adoption is safe even before the agent is rolled
out.
"""

from __future__ import annotations

from .announce import announce_alive
from .enrollment import EnrollmentHelperError, enroll_if_requested

__all__ = ["EnrollmentHelperError", "announce_alive", "enroll_if_requested"]
