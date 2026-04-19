# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Subprocess-based isolation for the MLIR pass pipeline.

Some FlyDSL/MLIR/LLVM C++ paths react to malformed input by calling
``llvm_unreachable``, ``assert``, or ``llvm::report_fatal_error`` — all of
which terminate the entire Python process via SIGABRT/SIGSEGV/SIGILL with
no Python-level exception.  Worse, the abort path may write partial ANSI
escape sequences to ``stderr`` and leak ROCm/HIP runtime resources, which
in interactive shells (Cursor IDE / SSH / tmux) can leave the terminal
unusable.

This module isolates every pass-pipeline run inside a *spawned* worker
subprocess.  Communication uses ``subprocess.Popen`` with explicit
``stdin``/``stdout``/``stderr`` PIPEs — this lets the parent capture all
diagnostic output (so nothing dirties the parent tty) and survive any
crash in the worker without losing context.

Public entry points:

* ``is_sandbox_enabled()`` — env-driven on/off switch.
* ``run_pass_pipeline_sandboxed(...)`` — runs a pipeline in a worker and
  returns a fresh ``Module`` parsed back into the caller's context.
* ``MlirCompilationCrashed`` — raised when the worker dies via signal.
* ``MlirCompilationTimeout`` — raised when the worker exceeds the budget.
"""

import io
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..utils import env

_DEFAULT_TIMEOUT_S = 600
_PROTOCOL_VERSION = 1


class MlirCompilationCrashed(Exception):
    """Worker died via signal (SIGABRT/SIGSEGV/SIGILL/...) — usually a bug
    in a native C++ pass (llvm_unreachable, assertion, segfault)."""

    def __init__(self, message: str, *, stderr: str = "", pipeline: str = ""):
        super().__init__(message)
        self.stderr = stderr
        self.pipeline = pipeline


class MlirCompilationTimeout(Exception):
    """Worker exceeded the timeout budget and was force-killed."""

    def __init__(self, message: str, *, stderr: str = "", pipeline: str = "", elapsed_s: float = 0.0):
        super().__init__(message)
        self.stderr = stderr
        self.pipeline = pipeline
        self.elapsed_s = elapsed_s


def is_sandbox_enabled() -> bool:
    return env.compile.sandbox


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"short read: wanted {n}, got {len(buf)}")
        buf.extend(chunk)
    return bytes(buf)


def _send_frame(stream, payload: bytes) -> None:
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def _recv_frame(stream) -> bytes:
    header = _read_exact(stream, 4)
    (size,) = struct.unpack("<I", header)
    return _read_exact(stream, size)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so worker diagnostics are safe to
    re-emit into a tty even if the worker was killed mid-write."""
    import re

    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


# ---------------------------------------------------------------------------
# Worker entry point — runs in a fresh interpreter via ``python -m``.
# ---------------------------------------------------------------------------


def _worker_main() -> int:
    """Read a pickled request from stdin, run the pipeline, write a
    pickled response to stdout.  All diagnostic output goes to stderr,
    which the parent will capture."""
    # Re-bind stdin/stdout to binary-only.  Anything the C++ side writes
    # to fd 1/2 still flows to the parent via PIPE; the worker uses
    # framed binary IO over stdin/stdout for control.
    raw_in = sys.stdin.buffer
    raw_out = sys.stdout.buffer

    try:
        request_bytes = _recv_frame(raw_in)
        request = pickle.loads(request_bytes)
    except Exception as exc:
        try:
            _send_frame(raw_out, pickle.dumps(("error", "string", f"failed to read request: {exc}")))
        except Exception:
            pass
        return 2

    pipeline: str = request["pipeline"]
    ir_text: str = request["ir_text"]
    enable_verifier: bool = request["enable_verifier"]
    enable_debug_info: bool = request["enable_debug_info"]
    print_after_all: bool = request.get("print_after_all", False)
    llvm_opts: Optional[Dict[str, Union[bool, int, str]]] = request.get("llvm_opts")

    try:
        from .._mlir import ir as mlir_ir
        from .._mlir.passmanager import PassManager
    except Exception as exc:
        _send_frame(raw_out, pickle.dumps(("error", "string", f"worker import failed: {exc}\n{traceback.format_exc()}")))
        return 2

    def _run() -> str:
        with mlir_ir.Context() as ctx:
            ctx.load_all_available_dialects()
            ctx.enable_multithreading(False)
            module = mlir_ir.Module.parse(ir_text, context=ctx)

            pm = PassManager.parse(pipeline, context=ctx)
            pm.enable_verifier(enable_verifier)
            if print_after_all:
                pm.enable_ir_printing(print_after_all=True)
            pm.run(module.operation)

            return module.operation.get_asm(enable_debug_info=enable_debug_info)

    try:
        if llvm_opts:
            from .llvm_options import llvm_options as _llvm_options

            with _llvm_options(llvm_opts):
                result_asm = _run()
        else:
            result_asm = _run()
    except BaseException as exc:
        try:
            from .._mlir.ir import MLIRError as _MLIRError
        except Exception:
            _MLIRError = None

        kind: str
        payload: Any
        if _MLIRError is not None and isinstance(exc, _MLIRError):
            diags = []
            try:
                for d in exc.error_diagnostics:
                    diags.append({"severity": str(d.severity), "message": str(d.message), "location": str(d.location)})
            except Exception:
                pass
            kind = "mlir_error"
            payload = {
                "message": str(getattr(exc, "message", "")),
                "args": tuple(str(a) for a in (exc.args or ())),
                "diagnostics": diags,
            }
        else:
            try:
                payload = pickle.dumps(exc)
                kind = "pickled"
            except Exception:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                payload = {
                    "type_module": type(exc).__module__,
                    "type_name": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb,
                }
                kind = "string"
        try:
            _send_frame(raw_out, pickle.dumps(("error", kind, payload)))
        except Exception:
            pass
        return 1

    try:
        _send_frame(raw_out, pickle.dumps(("ok", result_asm)))
    except Exception:
        return 3
    return 0


# ---------------------------------------------------------------------------
# Parent-side driver.
# ---------------------------------------------------------------------------


def _drain(stream, sink: io.BytesIO) -> None:
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            sink.write(chunk)
    except Exception:
        return


_WORKER_BOOTSTRAP = (
    "import sys; "
    "from flydsl.compiler.compile_sandbox import _worker_main; "
    "sys.exit(_worker_main())"
)


def _spawn_worker(timeout_s: float) -> subprocess.Popen:
    cmd = [sys.executable, "-u", "-c", _WORKER_BOOTSTRAP]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
        start_new_session=True,
    )


def run_pass_pipeline_sandboxed(
    module,
    pipeline: str,
    *,
    enable_verifier: bool = True,
    enable_debug_info: bool = False,
    print_after_all: bool = False,
    llvm_opts: Optional[Dict[str, Union[bool, int, str]]] = None,
    timeout_s: Optional[float] = None,
    capture_stderr_to: Optional[Path] = None,
):
    """Run ``pipeline`` on ``module`` inside an isolated worker process.

    On success, returns a freshly parsed ``Module`` belonging to the same
    MLIR context as the caller's ``module``.

    Failure modes:
      * pipeline raised a Python exception (e.g. ``MLIRError``,
        ``ValueError`` from a bad pipeline string) — the original exception
        type is preserved and re-raised in the parent.
      * worker died via signal — ``MlirCompilationCrashed`` with captured
        stderr (stripped of ANSI escapes for tty safety).
      * worker exceeded ``timeout_s`` — ``MlirCompilationTimeout``.
    """
    from .._mlir import ir as mlir_ir

    timeout = timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S

    ir_text = module.operation.get_asm(enable_debug_info=enable_debug_info)
    request = pickle.dumps(
        {
            "version": _PROTOCOL_VERSION,
            "pipeline": pipeline,
            "ir_text": ir_text,
            "enable_verifier": enable_verifier,
            "enable_debug_info": enable_debug_info,
            "print_after_all": print_after_all,
            "llvm_opts": llvm_opts,
        }
    )

    proc = _spawn_worker(timeout)
    stderr_buf = io.BytesIO()
    stdout_buf = io.BytesIO()
    stderr_thread = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf), daemon=True)
    stdout_thread = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf), daemon=True)
    stderr_thread.start()
    stdout_thread.start()

    write_failed: Optional[BaseException] = None
    try:
        proc.stdin.write(struct.pack("<I", len(request)))
        proc.stdin.write(request)
        proc.stdin.flush()
        proc.stdin.close()
    except BaseException as exc:
        write_failed = exc

    timed_out = False
    start = time.monotonic()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    except KeyboardInterrupt:
        try:
            proc.kill()
        finally:
            proc.wait(timeout=10)
        raise
    elapsed = time.monotonic() - start

    stderr_thread.join(timeout=5)
    stdout_thread.join(timeout=5)
    stderr_text = _strip_ansi(stderr_buf.getvalue().decode("utf-8", errors="replace"))
    stdout_bytes = stdout_buf.getvalue()

    if capture_stderr_to is not None and stderr_text.strip():
        try:
            capture_stderr_to.parent.mkdir(parents=True, exist_ok=True)
            capture_stderr_to.write_text(stderr_text, encoding="utf-8")
        except Exception:
            pass

    response: Optional[tuple] = None
    if stdout_bytes:
        try:
            stream = io.BytesIO(stdout_bytes)
            frame = _recv_frame(stream)
            response = pickle.loads(frame)
        except Exception:
            response = None

    if timed_out:
        raise MlirCompilationTimeout(
            f"MLIR compiler worker exceeded {timeout:.1f}s and was killed.\n"
            f"Pipeline: {pipeline}\n"
            f"--- worker stderr (last 2KB) ---\n{stderr_text[-2048:]}",
            stderr=stderr_text,
            pipeline=pipeline,
            elapsed_s=elapsed,
        )

    if write_failed is not None and response is None:
        raise MlirCompilationCrashed(
            f"Failed to send request to worker: {write_failed!r}\n"
            f"--- worker stderr ---\n{stderr_text[-2048:]}",
            stderr=stderr_text,
            pipeline=pipeline,
        )

    if response is None:
        rc = proc.returncode
        if rc is not None and rc < 0:
            sig_num = -rc
            sig_name = _CRASH_SIGNAL_NAMES.get(sig_num, f"signal {sig_num}")
            raise MlirCompilationCrashed(
                f"MLIR compiler worker was killed by {sig_name}.\n"
                f"This usually indicates a bug in a native pass "
                f"(llvm_unreachable, assertion, or segfault).\n"
                f"Pipeline: {pipeline}\n"
                f"--- worker stderr (last 2KB) ---\n{stderr_text[-2048:]}",
                stderr=stderr_text,
                pipeline=pipeline,
            )
        raise MlirCompilationCrashed(
            f"MLIR compiler worker exited with code {rc} but produced no response.\n"
            f"Pipeline: {pipeline}\n"
            f"--- worker stderr (last 2KB) ---\n{stderr_text[-2048:]}",
            stderr=stderr_text,
            pipeline=pipeline,
        )

    status = response[0]
    if status == "ok":
        result_asm = response[1]
        return mlir_ir.Module.parse(result_asm, context=module.context)

    if status == "error":
        kind, payload = response[1], response[2]
        if kind == "pickled":
            try:
                exc = pickle.loads(payload)
            except Exception:
                exc = None
            if exc is not None:
                if stderr_text:
                    try:
                        exc.add_note(f"worker stderr (last 2KB):\n{stderr_text[-2048:]}")
                    except Exception:
                        pass
                raise exc
            raise RuntimeError(
                f"MLIR pass pipeline failed (worker exception unpickle failed).\n"
                f"--- worker stderr (last 2KB) ---\n{stderr_text[-2048:]}"
            )

        if kind == "mlir_error":
            try:
                from .._mlir.ir import MLIRError
            except Exception:
                MLIRError = Exception  # type: ignore[assignment]

            class _RemoteMLIRError(MLIRError):
                pass

            msg = payload.get("message") or "MLIR pass pipeline failed"
            args = payload.get("args") or (msg,)
            diag_text = "\n".join(
                f"  [{d.get('severity', '?')}] {d.get('location', '?')}: {d.get('message', '')}"
                for d in payload.get("diagnostics") or []
            )
            full_msg = args[0] if args else msg
            if diag_text:
                full_msg = f"{full_msg}\n{diag_text}"
            if stderr_text:
                full_msg = f"{full_msg}\n--- worker stderr (last 2KB) ---\n{stderr_text[-2048:]}"
            raise _RemoteMLIRError(full_msg)

        if kind == "string":
            type_name = payload.get("type_name", "Exception") if isinstance(payload, dict) else "Exception"
            msg_text = payload.get("message", "") if isinstance(payload, dict) else str(payload)
            tb = payload.get("traceback", "") if isinstance(payload, dict) else ""
            full = f"MLIR pass pipeline failed: {type_name}: {msg_text}"
            if tb:
                full += f"\n{tb}"
            if stderr_text:
                full += f"\n--- worker stderr (last 2KB) ---\n{stderr_text[-2048:]}"
            raise RuntimeError(full)

        raise RuntimeError(f"MLIR pass pipeline failed (unknown error kind {kind!r}): {payload!r}")

    raise RuntimeError(f"Unknown worker response status: {status!r}")


_CRASH_SIGNAL_NAMES = {
    signal.SIGABRT: "SIGABRT (abort)",
    signal.SIGSEGV: "SIGSEGV (segmentation fault)",
    signal.SIGBUS: "SIGBUS (bus error)",
    signal.SIGFPE: "SIGFPE (floating point exception)",
    signal.SIGILL: "SIGILL (illegal instruction)",
    signal.SIGKILL: "SIGKILL (killed)",
    signal.SIGTERM: "SIGTERM (terminated)",
}


if __name__ == "__main__":
    sys.exit(_worker_main())
