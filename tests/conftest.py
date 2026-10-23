"""
Shared pytest fixtures and diagnostics for the VANNET test suite.

Design goals
------------
* **Isolation** - every test that drives the console gets its own
  :class:`SimulationState` through the application factory, so no test can be
  influenced by another test's background tasks, worker threads, replay
  buffers or trust state.
* **Deterministic cleanup** - every fixture tears down the FastAPI lifespan,
  the simulation loop, the broadcaster, the federated round task and the
  worker pool, and restores anything it patched.
* **Diagnostics** - slow tests are reported, and leftover worker threads or
  pending asyncio tasks are surfaced so a hang is always attributable.
"""

import asyncio
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from web.app import SimulationState, create_app  # noqa: E402


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
_TEST_DURATIONS = []
_SLOW_TESTS = []
_WORKER_THREAD_PREFIX = "vanet-fl-worker"
#: A test slower than this is flagged as a possible hang.
SLOW_TEST_SECONDS = 60.0


def pytest_runtest_logreport(report):
    """Record the duration of every test so slow ones can be reported."""
    if report.when == "call":
        _TEST_DURATIONS.append((report.nodeid, report.duration))
        # Flag a test that is slow enough to be a plausible hang: the report
        # identifies exactly which test is responsible.
        if report.duration > SLOW_TEST_SECONDS:
            _SLOW_TESTS.append((report.nodeid, report.duration))


def pytest_sessionfinish(session, exitstatus):
    """Print a diagnostics summary at the end of the session."""
    if not _TEST_DURATIONS:
        return
    print("\n" + "=" * 72)
    print("TEST DIAGNOSTICS: slowest tests")
    print("=" * 72)
    for nodeid, duration in sorted(_TEST_DURATIONS, key=lambda x: -x[1])[:15]:
        print("  %8.2fs  %s" % (duration, nodeid))

    if _SLOW_TESTS:
        print("\n  SLOW TESTS (over %.0f s - possible hang):" % SLOW_TEST_SECONDS)
        for nodeid, duration in _SLOW_TESTS:
            print("    %8.2fs  %s" % (duration, nodeid))

    alive = [t.name for t in threading.enumerate()
             if t.name.startswith(_WORKER_THREAD_PREFIX) and t.is_alive()]
    if alive:
        print("\n  WARNING: %d federated-worker thread(s) still alive at session end: %s"
              % (len(alive), alive))
    else:
        print("\n  No federated-worker threads left running at session end.")
    print("=" * 72)


def _live_worker_threads():
    return [t for t in threading.enumerate()
            if t.name.startswith(_WORKER_THREAD_PREFIX) and t.is_alive()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sim_state():
    """An isolated simulation instance with a fully managed worker pool."""
    state = SimulationState()
    try:
        yield state
    finally:
        state.request_worker_cancellation()
        state.shutdown_workers(wait=True, timeout=10.0)


@pytest.fixture
def console_client(sim_state):
    """A TestClient bound to an isolated application.

    Entering the context manager starts the FastAPI lifespan (the single
    simulation loop and the broadcaster); leaving it cancels and awaits every
    background task and shuts the worker pool down.
    """
    application = create_app(sim_state)
    with TestClient(application) as client:
        client.app_state = sim_state
        yield client
    # After the lifespan has shut down, no worker thread may remain.
    deadline = time.time() + 5.0
    while _live_worker_threads() and time.time() < deadline:
        time.sleep(0.05)


@pytest.fixture
def console_app(sim_state):
    """An isolated application (without starting its lifespan)."""
    return create_app(sim_state)


@pytest.fixture(autouse=True)
def _no_thread_leak_between_tests():
    """Fail fast if a federated worker thread survived the previous test."""
    yield
    alive = _live_worker_threads()
    if alive:
        # Give the worker a moment to finish its cooperative cancellation.
        deadline = time.time() + 5.0
        while _live_worker_threads() and time.time() < deadline:
            time.sleep(0.05)
        still = _live_worker_threads()
        if still:
            pytest.fail(
                "A federated worker thread outlived the test that started it: %s. "
                "This indicates a missing cleanup path." % [t.name for t in still]
            )
