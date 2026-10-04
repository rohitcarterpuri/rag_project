"""Console output helpers that avoid leaking machine-specific details.

Guarantees for all scripts that use this module:
- stdout/stderr are reconfigured to UTF-8 so emoji/status output cannot
  crash on non-UTF-8 Windows consoles (a crash dumps a traceback that
  exposes local usernames and project paths).
- Unhandled exceptions print a one-line, sanitized message instead of a
  traceback with local file paths. Set RAG_DEBUG=1 to opt back into full
  tracebacks for local debugging.
- `fail()` prints a controlled message and exits without a stack trace.
"""

import os
import re
import sys

# Anything that looks like an absolute filesystem path or a secret token
_REDACTIONS = [
    (re.compile(r"[A-Za-z]:[\\/][^\s'\"]+"), "[redacted-path]"),   # C:\Users\... / C:/...
    (re.compile(r"(?<![\w:])/(?:[^\s'\"]+/)+[^\s'\"]*"), "[redacted-path]"),  # /home/...
    (re.compile(r"\bgsk_[A-Za-z0-9]+"), "[redacted-key]"),        # Groq-style keys
    (re.compile(r"\bsk-[A-Za-z0-9]+"), "[redacted-key]"),         # sk-... keys
]


def _redact(text):
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, str(text))
    return text


def _sanitized_excepthook(exc_type, exc_value, traceback):
    if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        sys.__excepthook__(exc_type, exc_value, traceback)
        return
    debug = os.environ.get("RAG_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    if debug:
        sys.__excepthook__(exc_type, exc_value, traceback)
        return
    print(
        f"❌ Unexpected error ({_redact(exc_type.__name__)}: {_redact(exc_value)}). "
        "Set RAG_DEBUG=1 to see the full traceback.",
        file=sys.stderr,
    )


def install():
    """Enable safe console output. Call once, before any other output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.excepthook = _sanitized_excepthook


def fail(message):
    """Print a sanitized error (no tracebacks, no local paths) and exit."""
    print(f"❌ {message}", file=sys.stderr)
    sys.exit(1)
