"""The uniform logging contract: silent / compact / detailed, and one 'done.'.

Pins behavior that is easy to regress: level 0 must stay COMPLETELY silent (batch
sweeps depend on it), and a wrapper composing several stages must emit a single
trailing 'done.' rather than one per stage.
"""

import pytest

from deltatrace import _log


@_log.stage("doing the thing")
def _leaf(verbose=1):
    _log.detail("a detail")
    return 1


@_log.container()
def _wrapper(verbose=1):
    _leaf(verbose=verbose)
    _leaf(verbose=verbose)
    return 2


@pytest.mark.parametrize("level", [0, False, None])
def test_silent_levels_print_nothing(capsys, level):
    _wrapper(verbose=level)
    assert capsys.readouterr().out == ""


def test_compact_level_is_one_line_per_stage(capsys):
    _leaf(verbose=1)
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2                                  # step line + done.
    assert out[0].startswith("doing the thing ... finished (")
    assert "a detail" not in out[0]                       # details hidden at level 1
    assert out[-1] == "done."


@pytest.mark.parametrize("level", [2, True])
def test_detailed_level_shows_indented_details(capsys, level):
    _leaf(verbose=level)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "doing the thing ..."
    assert out[1] == "  a detail"                         # indented
    assert out[2].startswith("  finished (")
    assert out[-1] == "done."


def test_wrapper_emits_exactly_one_done(capsys):
    _wrapper(verbose=1)
    out = capsys.readouterr().out.splitlines()
    assert out.count("done.") == 1                        # not one per nested stage
    assert sum(line.startswith("doing the thing") for line in out) == 2


def test_detail_strips_leading_newlines(capsys):
    @_log.stage("s")
    def f(verbose=2):
        _log.detail("\nleading newline")

    f(verbose=2)
    assert "  leading newline" in capsys.readouterr().out.splitlines()


def test_resolve_maps_bools_and_clamps():
    assert _log.resolve(False) == 0 and _log.resolve(None) == 0
    assert _log.resolve(True) == 2
    assert _log.resolve(1) == 1
    assert _log.resolve(99) == 2 and _log.resolve(-5) == 0   # clamped


def test_verbose_never_changes_results(capsys):
    """The decorator must not perturb return values at any level."""
    assert [_leaf(verbose=v) for v in (0, 1, 2, True, False)] == [1] * 5
    capsys.readouterr()


# --------------------------------------------------------------------- failure paths
@_log.stage("doing the thing")
def _boom(verbose=1):
    raise ValueError("numerics blew up")


@pytest.mark.parametrize("level", [1, 2])
def test_a_raising_stage_reports_failed_not_finished(capsys, level):
    """Reporting 'finished' over a solve that raised is the one actively misleading
    output bug, so the step line must close with 'failed' and no 'done.' follow."""
    with pytest.raises(ValueError, match="numerics blew up"):
        _boom(verbose=level)
    out = capsys.readouterr().out
    assert "failed (" in out
    assert "finished (" not in out
    assert "done." not in out                    # 'done.' after a traceback reads as success


def test_a_raising_stage_still_terminates_its_line(capsys):
    """At level 1 the step line is written with end='' — it must be closed even on the
    error path, or the traceback lands mid-sentence."""
    with pytest.raises(ValueError):
        _boom(verbose=1)
    assert capsys.readouterr().out.endswith("\n")


def test_failure_is_silent_at_level_zero(capsys):
    with pytest.raises(ValueError):
        _boom(verbose=0)
    assert capsys.readouterr().out == ""


def test_exception_does_not_leak_the_level(capsys):
    """A raising stage must restore the module level, or a later detail() outside any
    stage would start printing."""
    with pytest.raises(ValueError):
        _boom(verbose=2)
    capsys.readouterr()
    _log.detail("must not appear")
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------ composing wrappers
def test_detail_works_in_a_container_body(capsys):
    """container() publishes the level too, so a composing wrapper can emit detail from
    its own body — previously dead at every verbosity (see pp.integrate)."""
    @_log.container()
    def compose(verbose=1):
        _log.detail("from the container body")
        _leaf(verbose=verbose)

    compose(verbose=2)
    out = capsys.readouterr().out.splitlines()
    assert "from the container body" in out          # flush-left: see next test
    assert out[-1] == "done."


def test_container_detail_is_not_indented_under_a_child_step(capsys):
    """A composing wrapper has no heading of its own, so indenting its details would file
    the wrapper's summary under whichever child step finished last."""
    @_log.container()
    def compose(verbose=1):
        _leaf(verbose=verbose)                       # prints its own indented detail
        _log.detail("wrapper summary")               # ...this belongs to the WRAPPER

    compose(verbose=2)
    out = capsys.readouterr().out.splitlines()
    assert "  a detail" in out                       # the child's detail IS indented
    assert "wrapper summary" in out                  # the wrapper's is not
    assert "  wrapper summary" not in out
    # and it must not be mistaken for part of the child's block
    assert out.index("wrapper summary") > out.index("  a detail")


def test_container_emits_no_step_line_of_its_own(capsys):
    @_log.container()
    def compose(verbose=1):
        _leaf(verbose=verbose)

    compose(verbose=1)
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2                                    # child's step line + done.
    assert out[0].startswith("doing the thing ... finished (")
    assert out[1] == "done."
    assert sum("finished (" in line for line in out) == 1   # only the child's


def test_container_is_stage_with_no_message(capsys):
    """The two decorators are one implementation; container() is the message-less form."""
    @_log.stage()
    def a(verbose=1):
        _leaf(verbose=verbose)

    @_log.container()
    def b(verbose=1):
        _leaf(verbose=verbose)

    a(verbose=1)
    first = capsys.readouterr().out
    b(verbose=1)
    second = capsys.readouterr().out
    assert first.count("finished (") == second.count("finished (") == 1
    assert first.endswith("done.\n") and second.endswith("done.\n")


def test_resolve_rejects_non_integer_verbose():
    """The old fallback mapped anything unparseable to 1 — NOISY — so verbose=[0],
    plainly asking for silence, printed."""
    for bad in ("loud", [2], [0], 1.7, object()):
        with pytest.raises(TypeError, match="verbose must be"):
            _log.resolve(bad)


def test_resolve_still_accepts_bools_and_integer_like_values():
    import numpy as np

    assert _log.resolve(np.int64(2)) == 2
    assert _log.resolve(2.0) == 2                         # integral float is fine
    assert _log.resolve(True) == 2 and _log.resolve(False) == 0


def test_a_bad_verbose_propagates_out_of_a_decorated_stage(capsys):
    """_log.stage's try only wraps sig.bind_partial, so resolve's TypeError reaches the
    caller instead of being swallowed into level 1."""
    with pytest.raises(TypeError, match="verbose must be"):
        _leaf(verbose="loud")
    assert capsys.readouterr().out == ""                  # and nothing was printed first
