"""Disposable JSON-lines worker with Linux seccomp syscall isolation.

Only inherited IPC descriptors remain usable. Opening files, sockets, processes,
executing programs and inspecting other processes are denied by the kernel.
"""

import json
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass

WORKER = r"""
import ctypes, json, resource, socket, sys, os
ipc_fd = os.dup(1)
os.dup2(2, 1)
transport_in = sys.stdin
transport_out = os.fdopen(ipc_fd, 'w')
if SANDBOX_UID is not None:
    os.setgroups([])
    os.setgid(SANDBOX_UID)
    os.setuid(SANDBOX_UID)
resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BYTES, RAM_BYTES))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
lib = ctypes.CDLL('libseccomp.so.2')
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
ctx = lib.seccomp_init(0x00050001)
for name in ('read','close','fstat','lseek','brk','mmap','munmap',
             'mprotect','mremap','futex','rt_sigaction','rt_sigprocmask','rt_sigreturn',
             'sigaltstack','clock_gettime','getrandom','getpid','gettid','getuid','geteuid','getgid','exit','exit_group'):
    n = lib.seccomp_syscall_resolve_name(name.encode())
    if n >= 0 and lib.seccomp_rule_add(ctx, 0x7fff0000, n, 0) != 0:
        raise RuntimeError('seccomp rule')
class ArgCmp(ctypes.Structure):
    _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_uint),
                ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]
lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                      ctypes.c_uint, ctypes.POINTER(ArgCmp)]
comparison = ArgCmp(0, 4, ipc_fd, 0)
if lib.seccomp_rule_add_array(ctx, 0x7fff0000,
        lib.seccomp_syscall_resolve_name(b'write'), 1, ctypes.byref(comparison)) != 0:
    raise RuntimeError('seccomp write rule')
if lib.seccomp_load(ctx) != 0: raise RuntimeError('seccomp unavailable')
def blocked(*a, **kw): raise PermissionError('network disabled')
socket.socket = blocked

def call(op, **args):
    transport_out.write(json.dumps(dict(version='editor_api.v1',op=op,args=args))+'\n')
    transport_out.flush()
    answer = json.loads(transport_in.readline())
    if 'error' in answer: raise RuntimeError(answer['error'])
    return answer['result']
code = json.loads(transport_in.readline())['code']
sys.stdout = sys.stderr
exec(compile(code, '<editor>', 'exec'), {'call':call})
transport_out.write(json.dumps({'done':True,'version':'editor_api.v1'})+'\n')
transport_out.flush()
"""


def audit(value):
    """No numeric arrays survive in the audit trail, including request arrays."""
    from wsp.editor.episode import digest

    if isinstance(value, list):

        def shape(v):
            return [len(v)] + (shape(v[0]) if v else []) if isinstance(v, list) else []

        return {"sha256": digest(value), "shape": shape(value)}
    if isinstance(value, dict):
        return {k: audit(v) for k, v in value.items()}
    return value


@dataclass
class ProposalResult:
    valid: bool
    error: str | None
    exit_code: int
    wall_seconds: float
    transcript: list


def run(
    code,
    api,
    *,
    wall_seconds=5,
    cpu_seconds=2,
    ram_bytes=256 * 1024**2,
    uid=None,
    require_separate_principal=False,
):
    if require_separate_principal and (uid is None or uid == os.getuid() or os.geteuid() != 0):
        raise ValueError("separate principal unavailable")
    if uid is not None and (uid <= 0 or os.geteuid() != 0):
        raise ValueError("unprivileged uid requires root supervisor")
    script = (
        WORKER.replace("SANDBOX_UID", repr(uid))
        .replace("CPU_SECONDS", str(cpu_seconds))
        .replace("RAM_BYTES", str(ram_bytes))
    )
    start, transcript = time.monotonic(), []
    proc = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={"LANG": "C.UTF-8"},
        close_fds=True,
        cwd="/",
    )
    api.deadline = start + wall_seconds
    done, error = False, None
    buffer = b""
    os.set_blocking(proc.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        proc.stdin.write((json.dumps({"code": code}) + "\n").encode())
        proc.stdin.flush()
        while not done:
            remaining = wall_seconds - (time.monotonic() - start)
            if remaining <= 0 or (b"\n" not in buffer and not selector.select(max(0, remaining))):
                raise TimeoutError("wall time limit")
            if b"\n" not in buffer:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise ValueError("worker exited before completion")
                buffer += chunk
                if len(buffer) > 1024 * 1024:
                    raise ValueError("IPC size limit")
                if b"\n" not in buffer:
                    continue
            line, buffer = buffer.split(b"\n", 1)
            request = json.loads(line)
            if request == {"done": True, "version": "editor_api.v1"}:
                done = True
                break
            transcript.append({"request": audit(request)})
            try:
                response = {"result": api.dispatch(request)}
            except Exception as exc:  # noqa: BLE001 - untrusted IPC must fail closed
                response = {"error": type(exc).__name__ + ": " + str(exc)}
            transcript[-1]["response"] = audit(response)
            proc.stdin.write((json.dumps(response) + "\n").encode())
            proc.stdin.flush()
        proc.wait(timeout=max(0.01, wall_seconds - (time.monotonic() - start)))
    except Exception as exc:  # noqa: BLE001 - untrusted IPC must fail closed
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        api.deadline = None
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        selector.close()
        proc.stdin.close()
        proc.stdout.close()
    return ProposalResult(
        done and proc.returncode == 0 and error is None,
        error,
        proc.returncode,
        time.monotonic() - start,
        transcript,
    )
