"""Supervisor-owned seccomp user-notification audit. Every denied syscall gets EPERM."""

import ctypes
import errno
import os
import select
import threading


class Data(ctypes.Structure):
    _fields_ = [
        ("nr", ctypes.c_int),
        ("arch", ctypes.c_uint32),
        ("ip", ctypes.c_uint64),
        ("args", ctypes.c_uint64 * 6),
    ]


class Request(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("pid", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("data", Data),
    ]


class Response(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("val", ctypes.c_int64),
        ("error", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class DenialAudit:
    def __init__(self, fd):
        self.fd, self.counts = fd, {}
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
        if lib.seccomp_api_get() < 5:
            self.error = "seccomp notification unsupported"
            return
        lib.seccomp_notify_receive.argtypes = [ctypes.c_int, ctypes.POINTER(Request)]
        lib.seccomp_notify_respond.argtypes = [ctypes.c_int, ctypes.POINTER(Response)]
        try:
            while not self.stop.is_set():
                if not select.select([self.fd], [], [], 0.05)[0]:
                    continue
                req = Request()
                status = lib.seccomp_notify_receive(self.fd, ctypes.byref(req))
                if status == -errno.ECANCELED and ctypes.get_errno() == errno.ENOENT:
                    return  # all supervised tasks have exited
                if status == -errno.ENOENT:
                    continue
                if status != 0:
                    raise OSError(-status, "seccomp receive")
                key = str(req.data.nr)
                self.counts[key] = self.counts.get(key, 0) + 1
                reply = Response(req.id, 0, -errno.EPERM, 0)
                status = lib.seccomp_notify_respond(self.fd, ctypes.byref(reply))
                if status not in (0, -errno.ENOENT):
                    raise OSError(-status, "seccomp respond")
        except Exception as exc:  # noqa: BLE001 - fail qualification on audit errors
            self.error = str(exc)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        os.close(self.fd)
        return {"enabled": True, "denied_syscalls": dict(self.counts), "error": self.error}
