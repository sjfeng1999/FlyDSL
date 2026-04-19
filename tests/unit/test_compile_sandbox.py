# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flydsl._mlir import ir
from flydsl._mlir.ir import MLIRError
from flydsl.compiler import compile_sandbox as cs
from flydsl.compiler.compile_sandbox import (
    MlirCompilationCrashed,
    MlirCompilationTimeout,
    is_sandbox_enabled,
    run_pass_pipeline_sandboxed,
)


def _make_trivial_module(ctx):
    return ir.Module.parse(
        """
        module {
          func.func @test(%arg0: i32) -> i32 {
            return %arg0 : i32
          }
        }
        """,
        context=ctx,
    )


def _spawn_aborting_worker(_timeout):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys, signal\n"
            "sys.stderr.write('worker: about to abort\\n'); sys.stderr.flush()\n"
            "os.kill(os.getpid(), signal.SIGABRT)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
        start_new_session=True,
    )


def _spawn_segfault_worker(_timeout):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, signal\n"
            "os.kill(os.getpid(), signal.SIGSEGV)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
        start_new_session=True,
    )


def _spawn_hanging_worker(_timeout):
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
        start_new_session=True,
    )


def _spawn_ansi_emitting_worker(_timeout):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys, signal\n"
            "sys.stderr.write('\\x1b[31mFATAL\\x1b[0m red\\nplain line\\n'); sys.stderr.flush()\n"
            "os.kill(os.getpid(), signal.SIGABRT)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
        start_new_session=True,
    )


class TestCompileSandboxBasic(unittest.TestCase):
    def test_sandbox_env_toggle(self):
        with mock.patch.dict(os.environ, {"FLYDSL_COMPILE_SANDBOX": "1"}):
            self.assertTrue(is_sandbox_enabled())
        with mock.patch.dict(os.environ, {"FLYDSL_COMPILE_SANDBOX": "0"}):
            self.assertFalse(is_sandbox_enabled())

    def test_successful_compilation(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            result = run_pass_pipeline_sandboxed(
                module, "builtin.module(canonicalize)", enable_verifier=True
            )
            self.assertIsNotNone(result)
            self.assertIn("@test", str(result))

    def test_result_belongs_to_caller_context(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            result = run_pass_pipeline_sandboxed(
                module, "builtin.module(canonicalize)", enable_verifier=True
            )
            self.assertEqual(result.context, ctx)

    def test_multiple_sequential_compilations(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            for _ in range(3):
                module = _make_trivial_module(ctx)
                result = run_pass_pipeline_sandboxed(
                    module, "builtin.module(canonicalize)", enable_verifier=True
                )
                self.assertIn("@test", str(result))


class TestCompileSandboxErrorTypes(unittest.TestCase):
    def test_pass_pipeline_parse_error_preserves_value_error(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with self.assertRaises(ValueError):
                run_pass_pipeline_sandboxed(
                    module,
                    "builtin.module(nonexistent-pass-xyz)",
                    enable_verifier=True,
                )

    def test_mlir_error_type_preserved(self):
        """A simulated MLIRError payload from the worker should reach the
        parent as an ``MLIRError`` instance (via the ``mlir_error`` reply
        kind), not a bare ``RuntimeError``."""
        import pickle
        import struct

        def _spawn_mlir_error_worker(_timeout):
            payload = pickle.dumps(
                (
                    "error",
                    "mlir_error",
                    {
                        "message": "synthetic verifier failure",
                        "args": ("synthetic verifier failure",),
                        "diagnostics": [
                            {
                                "severity": "DiagnosticSeverity.ERROR",
                                "message": "op is malformed",
                                "location": "loc(\"-\":1:1)",
                            }
                        ],
                    },
                )
            )
            framed = struct.pack("<I", len(payload)) + payload
            return subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; data = sys.stdin.buffer.read(); "
                    "sys.stdout.buffer.write({!r}); sys.stdout.buffer.flush()".format(framed),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            )

        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with mock.patch.object(cs, "_spawn_worker", _spawn_mlir_error_worker):
                with self.assertRaises(MLIRError) as cm:
                    run_pass_pipeline_sandboxed(
                        module, "builtin.module(canonicalize)"
                    )
            self.assertIn("synthetic verifier failure", str(cm.exception))
            self.assertIn("op is malformed", str(cm.exception))


class TestCompileSandboxCrashIsolation(unittest.TestCase):
    def test_sigabrt_caught_as_compilation_crashed(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with mock.patch.object(cs, "_spawn_worker", _spawn_aborting_worker):
                with self.assertRaises(MlirCompilationCrashed) as cm:
                    run_pass_pipeline_sandboxed(
                        module, "builtin.module(canonicalize)"
                    )
            self.assertIn("SIGABRT", str(cm.exception))
            self.assertEqual(cm.exception.pipeline, "builtin.module(canonicalize)")

    def test_sigsegv_caught_as_compilation_crashed(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with mock.patch.object(cs, "_spawn_worker", _spawn_segfault_worker):
                with self.assertRaises(MlirCompilationCrashed) as cm:
                    run_pass_pipeline_sandboxed(
                        module, "builtin.module(canonicalize)"
                    )
            self.assertIn("SIGSEGV", str(cm.exception))

    def test_parent_survives_then_compiles_again(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with mock.patch.object(cs, "_spawn_worker", _spawn_aborting_worker):
                with self.assertRaises(MlirCompilationCrashed):
                    run_pass_pipeline_sandboxed(
                        module, "builtin.module(canonicalize)"
                    )

        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            result = run_pass_pipeline_sandboxed(
                module, "builtin.module(canonicalize)", enable_verifier=True
            )
            self.assertIn("@test", str(result))


class TestCompileSandboxStderrCapture(unittest.TestCase):
    def test_stderr_captured_on_crash(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with mock.patch.object(cs, "_spawn_worker", _spawn_aborting_worker):
                with self.assertRaises(MlirCompilationCrashed) as cm:
                    run_pass_pipeline_sandboxed(
                        module, "builtin.module(canonicalize)"
                    )
            self.assertIn("worker: about to abort", cm.exception.stderr)

    def test_stderr_ansi_stripped(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with mock.patch.object(cs, "_spawn_worker", _spawn_ansi_emitting_worker):
                with self.assertRaises(MlirCompilationCrashed) as cm:
                    run_pass_pipeline_sandboxed(
                        module, "builtin.module(canonicalize)"
                    )
            stderr = cm.exception.stderr
            self.assertIn("FATAL", stderr)
            self.assertNotIn("\x1b[", stderr)

    def test_stderr_written_to_capture_file_on_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "subdir" / "stderr.log"
            with ir.Context() as ctx:
                ctx.load_all_available_dialects()
                module = _make_trivial_module(ctx)
                with mock.patch.object(cs, "_spawn_worker", _spawn_aborting_worker):
                    with self.assertRaises(MlirCompilationCrashed):
                        run_pass_pipeline_sandboxed(
                            module,
                            "builtin.module(canonicalize)",
                            capture_stderr_to=log_path,
                        )
                self.assertTrue(log_path.exists())
                self.assertIn("worker: about to abort", log_path.read_text())


class TestCompileSandboxTimeout(unittest.TestCase):
    def test_timeout_distinct_from_crash(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with mock.patch.object(cs, "_spawn_worker", _spawn_hanging_worker):
                with self.assertRaises(MlirCompilationTimeout) as cm:
                    run_pass_pipeline_sandboxed(
                        module, "builtin.module(canonicalize)", timeout_s=0.5
                    )
            self.assertGreaterEqual(cm.exception.elapsed_s, 0.4)
            self.assertEqual(cm.exception.pipeline, "builtin.module(canonicalize)")


class TestCompileSandboxIRPrinting(unittest.TestCase):
    def test_print_after_all_routes_to_worker_stderr(self):
        """When ``print_after_all=True``, the worker's PassManager IR
        printing goes to stderr, which the sandbox captures."""
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = _make_trivial_module(ctx)
            with tempfile.TemporaryDirectory() as tmp:
                log_path = Path(tmp) / "ir_printing.log"
                run_pass_pipeline_sandboxed(
                    module,
                    "builtin.module(canonicalize)",
                    enable_verifier=True,
                    print_after_all=True,
                    capture_stderr_to=log_path,
                )
                self.assertTrue(log_path.exists(), "stderr log should be created when print_after_all is on")
                txt = log_path.read_text()
                self.assertIn("@test", txt)


if __name__ == "__main__":
    unittest.main()
