"""Uniform step logging for the public DeltaTrace stages.

One scheme everywhere: each stage prints ``message ... finished (t)``, and the outermost
call closes with a trailing ``done.``. Verbosity is a 0/1/2 scale -- silent, compact, or
compact plus indented detail -- resolved from a caller's ``verbose`` via :func:`resolve`.

Decorate a public stage with :func:`stage` (or :func:`container` for a wrapper that only
composes other stages) and call :func:`detail` inside it for level-2 output.
"""

from __future__ import annotations

import functools
import inspect
import sys
import time
from contextlib import contextmanager

_depth = 0          # nesting depth, so only the outermost task prints "done."
_level = 0          # level of the step currently executing (for detail())
_in_step = False    # is a step heading open? decides whether detail() indents


def resolve(verbose) -> int:
    """Map ``verbose`` (bool or int) onto the 0/1/2 scale.

    ``None``/``False`` mean silent and ``True`` means full detail. Raises ``TypeError`` if
    ``verbose`` isn't integer-like, rather than silently falling back to level 1.

    Parameters
    ----------
    verbose : bool or int
        Caller's verbosity argument.

    Returns
    -------
    int
        0, 1, or 2.
    """
    if verbose is None or verbose is False:
        return 0
    if verbose is True:
        return 2
    try:
        level = int(verbose)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"verbose must be a bool or an int in 0..2, got {verbose!r}"
        ) from exc
    # int() truncates 1.7 and accepts numpy floats, so require the round trip to hold.
    if level != verbose:
        raise TypeError(
            f"verbose must be a bool or an int in 0..2, got {verbose!r}"
        )
    return max(0, min(2, level))


def _is_terminal() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _fmt(seconds: float) -> str:
    if seconds >= 60:
        total_sec = round(seconds)
        m, s = divmod(total_sec, 60)
        if m >= 60:
            h, m = divmod(m, 60)
            return f"{int(h)} h {int(m)} m"
        return f"{int(m)} m {int(s)} s"
    if seconds >= 1:
        return f"{seconds:.1f} s"
    return f"{seconds:.2f} s"


def detail(msg: str) -> None:
    """Print a detail line, shown only at level 2.

    Indented under the current step heading, or flush-left when no step is open (as in a
    composing wrapper). Surrounding whitespace is stripped.
    """
    if _level >= 2:
        print(f"  {msg.strip()}" if _in_step else msg.strip(), flush=True)


def wants_detail() -> bool:
    """True when :func:`detail` would actually print.

    Guard *expensive* detail computation with this; ``detail`` itself already
    suppresses output, but its f-string arguments are still evaluated.
    """
    return _level >= 2


@contextmanager
def step(message: str, verbose):
    """Time one stage and report it uniformly. ``message`` supplies its own verb.

    On success the line closes with ``finished (t)``; if the body raises it closes with
    ``failed (t)`` instead and the exception propagates untouched.
    """
    global _level, _in_step
    lvl = resolve(verbose)
    prev, _level = _level, lvl
    prev_in, _in_step = _in_step, True
    try:
        if lvl == 0:
            yield
            return
        # Not flushed in a notebook: a flushed half is saved as an output of its own (nbclient,
        # papermill), which viewers draw on its own line; the kernel's periodic flush still
        # shows a long step while it runs. A terminal gets it straight away.
        print(f"{message} ...{'' if lvl >= 2 else ' '}", end="\n" if lvl >= 2 else "",
              flush=_is_terminal())
        t0 = time.perf_counter()
        try:
            yield
        # BaseException, not Exception: a KeyboardInterrupt must still end the `end=""` line.
        except BaseException:
            _close(f"failed ({_fmt(time.perf_counter() - t0)})", lvl)
            raise
        else:
            _close(f"finished ({_fmt(time.perf_counter() - t0)})", lvl)
    finally:
        _level = prev
        _in_step = prev_in


def _close(what: str, lvl: int) -> None:
    """Terminate a step line, indented at level 2 to sit under its own heading."""
    print(f"  {what}" if lvl >= 2 else what, flush=True)


@contextmanager
def task(verbose):
    """Wrap a public entry point; emits the trailing ``done.``.

    Nesting-aware: only the outermost ``task`` prints it, and nothing prints if the body
    raises. Also publishes the level, so :func:`detail` works in a composing wrapper's own
    body and not only inside a :func:`step`.
    """
    global _depth, _level
    lvl = resolve(verbose)
    _depth += 1
    prev, _level = _level, lvl
    ok = True
    try:
        yield lvl
    except BaseException:
        ok = False
        raise
    finally:
        _depth -= 1
        _level = prev
        if ok and _depth == 0 and lvl > 0:
            print("done.", flush=True)


def stage(message=None):
    """Decorate a public stage so it logs uniformly.

    Reads the decorated function's own ``verbose`` argument (bool or 0/1/2) to control
    output. ``message=None`` marks a wrapper that only *composes* other stages (``integrate``,
    ``trajectory``): it emits no step line of its own, so the output is a flat list of its
    children's steps followed by a single ``done.``. Use the :func:`container` alias to say
    so at the call site. ``message`` may also be a callable that is given the call's bound
    arguments and returns the line to print.
    """
    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                verbose = bound.arguments.get("verbose", 1)
                msg = message(bound.arguments) if callable(message) else message
            except TypeError:                     # let the real call raise properly
                verbose = 1
                msg = message if not callable(message) else "working"
            with task(verbose) as lvl:
                if msg is None:                   # composing wrapper: no step line
                    return fn(*args, **kwargs)
                with step(msg, lvl):
                    return fn(*args, **kwargs)

        return wrapper

    return deco


def container():
    """Alias for ``stage()`` with no message: see :func:`stage`.

    States the intent at the call site, where a bare ``@_log.stage()`` would look like a
    missing argument.
    """
    return stage(None)
