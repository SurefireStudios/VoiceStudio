"""Every sidecar's generate deadline outlasts the job budget it was granted (#2103).

#1611 raised IndexTTS's deadline because a healthy synthesis was being killed at
60s. That fixed the reported engine and left the class default alone, so four
more engines — confucius4, dots_tts, moss_tts_v15, supertonic3 — inherited the
same 60s and were killed the same way.

60s is the ``health_check`` ping budget. Inheriting it as a *generation*
deadline puts the sidecar watchdog five to ten times below
``model_manager.generate_timeout_s`` (300s accelerated, 600s CPU), so the
watchdog reclaims a sidecar the caller still considers well inside its budget.
Every engine that overrode the hook picked 300s..900s, i.e. at or above the
accelerated budget; the four that stayed silent are the whole bug.

The invariant below is what keeps a new engine from re-entering that state by
omission, which is the part #1611 could not do by fixing one engine.
"""
import pytest

from engines.omnivoice_subprocess import OmniVoiceSubprocessBackend
from services.model_manager import CPU_JOB_TIMEOUT_S, GPU_JOB_TIMEOUT_S
from services.subprocess_backend import (
    GENERATE_RECV_TIMEOUT_S,
    RECV_TIMEOUT_S,
    SubprocessBackend,
)
from services.tts_backend import get_backend_class, list_backends


# The engines named in #2103 that inherited the ping budget. Listed explicitly
# so the regression is legible even if the registry is reorganised later.
REGRESSED_ENGINE_IDS = ("confucius4-tts", "dots-tts", "moss-tts-v15", "supertonic3")


def _subprocess_backend_classes():
    """Every SubprocessBackend the registry can hand a user, by id."""
    found = {}
    for row in list_backends(include_hidden=True):
        try:
            cls = get_backend_class(row["id"])
        except Exception:
            continue  # an engine whose optional import is absent cannot be dispatched
        if isinstance(cls, type) and issubclass(cls, SubprocessBackend):
            found[row["id"]] = cls
    return found


def test_ping_budget_and_generate_budget_are_separate_constants():
    # The bug was one constant serving both roles. A ping must stay fast; a
    # generation must not be cut off at a ping's deadline.
    assert RECV_TIMEOUT_S == 60.0
    assert GENERATE_RECV_TIMEOUT_S > RECV_TIMEOUT_S


def test_default_generate_deadline_covers_the_cpu_job_budget():
    # Lockstep with model_manager: raising either budget there without raising
    # this one re-opens #2103 for every engine that does not override.
    # Imported rather than duplicated so the two cannot drift silently.
    assert GENERATE_RECV_TIMEOUT_S >= CPU_JOB_TIMEOUT_S
    assert SubprocessBackend.recv_timeout_s == GENERATE_RECV_TIMEOUT_S


@pytest.mark.parametrize("engine_id", REGRESSED_ENGINE_IDS)
def test_regressed_engines_no_longer_inherit_the_ping_budget(engine_id):
    cls = _subprocess_backend_classes().get(engine_id)
    if cls is None:
        pytest.skip(f"{engine_id} is not registered in this build")
    # Read through an instance: several engines expose the hook as a property.
    assert cls.__new__(cls).recv_timeout_s > RECV_TIMEOUT_S


def test_no_registered_sidecar_undercuts_the_accelerated_job_budget():
    """The class-level guard #1611 was missing.

    A new SubprocessBackend that simply does not think about ``recv_timeout_s``
    now inherits a deadline that already satisfies this; one that overrides it
    with something too small fails here rather than in a user's generation.
    """
    too_short = {}
    for engine_id, cls in _subprocess_backend_classes().items():
        try:
            deadline = cls.__new__(cls).recv_timeout_s
        except Exception:
            continue  # a property needing real instance state is exercised elsewhere
        if deadline < GPU_JOB_TIMEOUT_S:
            too_short[engine_id] = deadline
    assert not too_short, (
        "these sidecars would be killed before their own job budget expires: "
        f"{too_short} (accelerated budget is {GPU_JOB_TIMEOUT_S:g}s)"
    )


# ── the deadline has to appear in the error the caller sees (#2103) ─────────

# Wedges on the first synthesize, so the parent's watchdog is the only thing
# that can end the request — the exact shape the #1611 and #2103 reporters hit.
WEDGING_SIDECAR = r'''
import sys, json, struct, time

def _send(o):
    b = json.dumps(o, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack("!I", len(b)) + b)
    sys.stdout.buffer.flush()

_send({"op": "ready", "engine": "omnivoice-subprocess", "sample_rate": 24000})
print("sidecar still alive, just slow", file=sys.stderr, flush=True)
while True:
    time.sleep(1)
'''


def test_timeout_error_names_the_deadline_instead_of_blaming_the_pipe(
    tmp_path, monkeypatch,
):
    """#2103's second half: the watchdog's own deadline reached the user.

    Before this, a kill and a crash both raised "sidecar closed pipe
    mid-generate", so the one fact that explains the failure — that
    VoiceStudio stopped the sidecar on its own deadline — appeared only in the
    backend log, and reporters reasonably concluded the engine had crashed.
    """
    script = tmp_path / "wedging_sidecar.py"
    script.write_text(WEDGING_SIDECAR)
    monkeypatch.setattr(
        OmniVoiceSubprocessBackend, "sidecar_script", classmethod(lambda cls: script),
    )
    # 2s so the test is fast; the property floors env overrides at 30s, so set
    # the attribute the base actually reads (as the existing wedge test does).
    monkeypatch.setattr(
        OmniVoiceSubprocessBackend, "recv_timeout_s", property(lambda self: 2.0),
    )
    backend = OmniVoiceSubprocessBackend()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            backend.generate("anything")
    finally:
        backend.shutdown()

    message = str(excinfo.value)
    assert "2s" in message, message          # the deadline that ended it
    assert "stopped it" in message, message  # who ended it, not "it closed"
    assert "closed pipe" not in message, message
    # #2026's stderr tail is carried on this path too, so a sidecar that did
    # say something before the kill is not silenced by the timeout.
    assert "still alive" in message, message
