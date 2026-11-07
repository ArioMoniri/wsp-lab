"""Oriented local response chart: per-tensor tokens and augmented-StatNN columns (PREREG A5.8.2-3).

``chart_tokens`` turns one chain step into one token per edited tensor. The edit is the realised
increment ``D = successor - predecessor``: both states are stored bf16, the successor is
``bf16(base + cumulative entry delta)``, and the subtraction is done in fp32. The displacement is
``E = W_pred - W_base`` (predecessor minus base, same convention). Matrix tensors get coordinate
blocks ``P_out^T X P_in`` for ``X`` in ``{W, E, D}``, the signed elementwise contractions of the
edit block with the predecessor, base and displacement blocks, raw inner products, norms, bf16
rounding loss and a spectral tail. Non-matrix tensors (1-D norms, biases, mixer state such as
``A_log`` or the 3-D ``conv1d``) get a reduced token: ``edit_norm`` and ``displacement_norm`` only,
no contractions; they are never reshaped into a matrix.

``to_columns`` flattens tokens onto the fixed slot layout of the ``wsp.zoo.arch`` registry
(``Arch.chart_roles`` for matrix slots, ``Arch.chart_vector_roles`` for vector slots): every
``(layer, role)`` pair registered for the configured ``layer_types``, ordered by layer then role
string. Each slot carries its token scalars plus a presence flag; an absent slot is all zero with
presence 0, and a coordinate mask marks which entries were observed. Columns depend only on the
schema and the set of tokens, never on token order.

Chair defaults, each **chair default, Board A may change before freeze** (PREREG A5.8.2-3):
projector dimension 32 per side; per-tensor projector seed
``SeedSequence([20261107, crc32(tensor_name)])`` (first uint64 state); Gaussian frames
orthonormalised by QR with a positive-diagonal sign convention; slots from the registry ordered by
layer then role, absent slots zero with presence 0; the reduced non-matrix token; the
realised-increment convention above.

``CHART_SCHEMA_VERSION`` bump rule: increment it in the same commit as any change that can alter
``chart_tokens`` or ``to_columns`` output for identical inputs, including field set, field order or
sizes, column names, slot registry entries (``chart_roles`` / ``chart_vector_roles``), slot order,
padding/mask/presence semantics, projector dimension default, seed derivation, frame construction
or the increment/displacement conventions. Pure refactors with bit-identical output do not bump it.
Cached columns are valid only for the version (and ``schema_hash``) they were written with.
"""

import hashlib
import json
import re
import zlib

import numpy as np
import torch

from wsp.zoo.apply import load_entry_delta
from wsp.zoo.arch import ARCHS, split_key

CHART_SCHEMA_VERSION = 1


def schema_hash(schema):
    """Hash a JSON schema independently of dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_seed(name):
    """Derive a tensor-specific seed."""
    return int(
        np.random.SeedSequence([20261107, zlib.crc32(name.encode())]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def slot_layout(architecture, layer_types):
    """``(layer, role, is_matrix)`` for every registered slot, ordered by layer then role."""
    arch = ARCHS[architecture]
    matrix, vector = arch.chart_roles, arch.chart_vector_roles
    if not matrix:
        raise ValueError("architecture has no registered chart roles")
    layout = []
    for i, kind in enumerate(layer_types):
        if kind not in matrix and kind not in vector:
            raise ValueError(f"layer {i}: no chart roles for layer type {kind!r}")
        m = {*matrix.get("all", ()), *matrix.get(kind, ())}
        v = {*vector.get("all", ()), *vector.get(kind, ())}
        if m & v:
            raise ValueError(f"roles registered as both matrix and vector: {sorted(m & v)}")
        layout.extend((i, role, role in m) for role in sorted(m | v))
    return layout


def architecture_slots(architecture, layer_types):
    """Enumerate registered ``(layer, role)`` slots in layer and role order."""
    return [(layer, role) for layer, role, _ in slot_layout(architecture, layer_types)]


def block_projectors(out_dim, in_dim, k, seed):
    """Generate independent Gaussian QR frames with a positive diagonal."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def frame(n):
        q, r = torch.linalg.qr(torch.randn(n, min(k, n), generator=gen))
        signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
        return q * signs

    return frame(out_dim), frame(in_dim)


def contractions(D, W_pred, W_base, P_out, P_in):
    """Return signed coordinate-block contractions and unnormalised diagnostics."""
    d, w, b = (x.detach().to(device="cpu", dtype=torch.float32) for x in (D, W_pred, W_base))
    e = w - b
    pd = P_out.T @ d @ P_in
    result = {"projected_edit": pd}
    for name, x in (("predecessor", w), ("base", b), ("displacement", e)):
        projected = P_out.T @ x @ P_in
        result["projected_" + name] = projected
        result[name] = (pd * projected).flatten()
    singular = torch.linalg.svdvals(d)
    total = singular.square().sum()
    result.update(
        raw_predecessor=(d * w).sum(),
        raw_displacement=(d * e).sum(),
        edit_norm=torch.linalg.vector_norm(d),
        displacement_norm=torch.linalg.vector_norm(e),
        rounding_loss=torch.linalg.vector_norm((w + d).bfloat16().float() - w - d),
        spectral_tail=singular[8:].square().sum() / total if total > 0 else torch.tensor(0.0),
        spectral_present=bool(total > 0),
    )
    return result


def tensor_slot(name):
    """``(layer, role)`` of a layer tensor: weights drop ``.weight``, others keep the full path."""
    parsed = split_key(name)
    if parsed is not None:
        return parsed
    match = re.fullmatch(r"model\.(?:language_model\.)?layers\.(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"no layer role for {name}")
    return int(match.group(1)), match.group(2)


def chart_tokens(entry_dir, predecessor_state, base_state, schema):
    """Construct tokens from cumulative base-relative entry and stored predecessor states."""
    delta = load_entry_delta(entry_dir, base_state)
    k = schema.get("projector_dim", 32)
    tokens = []
    for name in delta:
        layer, role = tensor_slot(name)
        w = predecessor_state[name]
        base = base_state[name]
        if w.dtype != torch.bfloat16 or base.dtype != torch.bfloat16:
            raise ValueError("stored states must be bf16")
        successor = (base.float() + delta[name].float()).bfloat16()
        d = successor.float() - w.float()
        if d.ndim == 2:
            p, q = block_projectors(*d.shape, k, tensor_seed(name))
            token = contractions(d, w, base, p, q)
        else:
            token = {
                "edit_norm": torch.linalg.vector_norm(d),
                "displacement_norm": torch.linalg.vector_norm(w.float() - base.float()),
            }
        token.update(name=name, layer=layer, role=role, shape=list(d.shape))
        tokens.append(token)
    return sorted(tokens, key=lambda t: (t["layer"], t["role"], t["name"]))


MATRIX_FIELDS = ("predecessor", "base", "displacement")
MATRIX_SCALARS = (
    "raw_predecessor",
    "raw_displacement",
    "edit_norm",
    "displacement_norm",
    "rounding_loss",
    "spectral_tail",
    "spectral_present",
)
VECTOR_SCALARS = ("edit_norm", "displacement_norm")


def to_columns(tokens, schema):
    """Flatten fixed registered slots with coordinate and tensor presence masks."""
    k = schema.get("projector_dim", 32)
    layout = slot_layout(schema["architecture"], schema["layer_types"])
    by_slot = {(t["layer"], t["role"]): t for t in tokens}
    kinds = {(layer, role): is_matrix for layer, role, is_matrix in layout}
    if len(by_slot) != len(tokens) or set(by_slot) - set(kinds):
        raise ValueError("duplicate or unknown tensor slot")
    for slot, token in by_slot.items():
        if (len(token["shape"]) == 2) != kinds[slot]:
            raise ValueError(f"tensor shape {token['shape']} does not fit slot kind of {slot}")
    matrix_fields = [(name, k * k) for name in MATRIX_FIELDS] + [(n, 1) for n in MATRIX_SCALARS]
    vector_fields = [(name, 1) for name in VECTOR_SCALARS]
    values, names, mask = [], [], []
    for layer, role, is_matrix in layout:
        token = by_slot.get((layer, role))
        for field, size in matrix_fields if is_matrix else vector_fields:
            x = np.asarray(
                token[field] if token is not None and field in token else [], dtype=np.float32
            ).reshape(-1)
            if x.size > size:
                raise ValueError("field exceeds slot size")
            if field in MATRIX_FIELDS and x.size:
                rows, cols = min(k, token["shape"][0]), min(k, token["shape"][1])
                padded = np.zeros((k, k), dtype=np.float32)
                present = np.zeros((k, k), dtype=bool)
                padded[:rows, :cols] = x.reshape(rows, cols)
                present[:rows, :cols] = True
                values.extend(padded.ravel())
                mask.extend(present.ravel())
            else:
                values.extend(np.pad(x, (0, size - x.size)))
                mask.extend([True] * x.size + [False] * (size - x.size))
            names.extend(f"{layer}:{role}:{field}:{i}" for i in range(size))
        values.append(float(token is not None))
        mask.append(True)
        names.append(f"{layer}:{role}:presence")
    return np.asarray(values, dtype=np.float32), names, np.asarray(mask, dtype=bool)
