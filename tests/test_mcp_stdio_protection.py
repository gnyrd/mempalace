"""Regression tests for issue #225 — MCP stdio protection.

The MCP protocol multiplexes JSON-RPC over stdio. Stdout MUST carry only
valid JSON-RPC messages. Several transitive deps (chromadb → onnxruntime,
posthog telemetry) print banners and warnings to stdout — sometimes at
the C level — which broke Claude Desktop's JSON parser on Windows.

The fix in mcp_server.py redirects stdout → stderr at both the Python
and file-descriptor level during module import, then restores the real
stdout in main() before entering the protocol loop.
"""

import subprocess
import sys
import textwrap


def test_module_import_redirects_stdout_to_stderr():
    """At import time, sys.stdout must point at sys.stderr so any stray
    print() from a transitive dependency is sent to stderr."""
    code = textwrap.dedent(
        """
        import sys
        original_stdout = sys.stdout
        from mempalace import mcp_server
        assert sys.stdout is sys.stderr, (
            f"Expected sys.stdout to be redirected to sys.stderr, "
            f"got: {sys.stdout!r}"
        )
        assert mcp_server._REAL_STDOUT is original_stdout, (
            "mcp_server._REAL_STDOUT must hold the original stdout"
        )
        print("OK", file=sys.stderr)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"


def test_restore_stdout_returns_real_stdout():
    """_restore_stdout() must reassign sys.stdout to the original handle
    so main() can write JSON-RPC responses to the real stdout."""
    code = textwrap.dedent(
        """
        import sys
        original_stdout = sys.stdout
        from mempalace import mcp_server
        assert sys.stdout is sys.stderr
        mcp_server._restore_stdout()
        assert sys.stdout is original_stdout, (
            f"After _restore_stdout(), sys.stdout must be the original; "
            f"got: {sys.stdout!r}"
        )
        mcp_server._restore_stdout()  # idempotent
        print("OK", file=sys.stderr)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"


def test_mcp_server_no_stdout_noise_on_clean_exit():
    """`python -m mempalace.mcp_server` with empty stdin must produce
    nothing on stdout. Empty input → readline() returns '' → main()
    breaks out cleanly. Any stdout content here would corrupt the
    JSON-RPC stream in real use."""
    proc = subprocess.run(
        [sys.executable, "-m", "mempalace.mcp_server"],
        input=b"",
        capture_output=True,
        timeout=60,
    )
    assert (
        proc.stdout == b""
    ), f"stdout must be empty before the first JSON-RPC response, but got: {proc.stdout!r}"


def test_emit_jsonrpc_survives_fd1_clobber():
    """JSON-RPC responses must reach the original stdout pipe even after a
    transitive C-level dependency re-clobbers fd 1.

    Regression for the write-guard v2 rollout incident: chromadb's
    onnxruntime + CoreMLExecutionProvider init at first chromadb access
    silently dup2(2, 1)'d fd 1. Every subsequent ``sys.stdout.write``
    landed on stderr instead of the parent pipe. From the MCP client's
    view, the call hung indefinitely (observed: 11+ minutes before the
    transport gave up with ``Connection closed``).

    Defense (mcp_server._emit_jsonrpc): keep ``_REAL_STDOUT_FD`` open as a
    persistent reference to the original stdout pipe and write each
    response via ``os.write(_REAL_STDOUT_FD, ...)`` — bypasses fd 1
    entirely so a clobber is harmless.

    This test simulates the clobber explicitly via ``os.dup2(2, 1)`` after
    server startup and asserts the JSON-RPC frame still lands on the
    parent's stdout, not its stderr.
    """
    code = textwrap.dedent(
        """
        import os
        from mempalace import mcp_server
        mcp_server._restore_stdout()
        # Simulate chromadb/onnxruntime C-level fd-1 clobber.
        os.dup2(2, 1)
        mcp_server._emit_jsonrpc(
            {"jsonrpc": "2.0", "id": 42, "result": {"ok": True}}
        )
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"child exited non-zero. stdout: {proc.stdout!r} stderr: {proc.stderr!r}"
    )
    assert b'"id": 42' in proc.stdout, (
        "_emit_jsonrpc must write to the original stdout pipe even after fd 1 "
        f"is clobbered. stdout: {proc.stdout!r} stderr: {proc.stderr!r}"
    )
    assert b'"id": 42' not in proc.stderr, (
        "JSON-RPC frame leaked to stderr — fd-clobber defense is broken. "
        f"stderr: {proc.stderr!r}"
    )


def test_real_stdout_fd_persists_after_restore_for_emit_path():
    """``_restore_stdout`` must keep ``_REAL_STDOUT_FD`` open. ``_emit_jsonrpc``
    relies on the saved fd to bypass fd 1 (which can be re-clobbered by
    transitive deps mid-session). Closing the fd would force the emitter
    onto its sys.stdout fallback — defeating the defense.
    """
    code = textwrap.dedent(
        """
        import os
        from mempalace import mcp_server
        saved = mcp_server._REAL_STDOUT_FD
        assert saved is not None, "_REAL_STDOUT_FD should be set at import"
        mcp_server._restore_stdout()
        # After restore, the saved fd must still be valid (not closed) so
        # _emit_jsonrpc can keep using it.
        assert mcp_server._REAL_STDOUT_FD == saved, (
            "_REAL_STDOUT_FD changed across _restore_stdout() — emitter relies "
            "on a stable saved fd"
        )
        # fstat will raise OSError(EBADF) if the fd was closed.
        os.fstat(mcp_server._REAL_STDOUT_FD)
        print("OK", file=__import__("sys").stderr)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
