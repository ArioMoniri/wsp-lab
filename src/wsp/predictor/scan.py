"""Chunked selective scan implemented with torch operations."""

import torch


def selective_scan(u, dt, a, b, c, d, chunk_size=64):
    """Evaluate the selective recurrence, carrying state between chunks."""
    h = u.new_zeros(u.shape[0], u.shape[2], a.shape[1])
    outputs = []
    for start in range(0, u.shape[1], chunk_size):
        sl = slice(start, start + chunk_size)
        log_decay = dt[:, sl, :, None] * a
        # Subdivide the chunk with an associative scan. This avoids division by
        # tiny cumulative products while retaining log-space decay products.
        values = dt[:, sl, :, None] * b[:, sl, None, :] * u[:, sl, :, None]
        decay = log_decay
        offset = 1
        while offset < values.shape[1]:
            values = torch.cat(
                (
                    values[:, :offset],
                    values[:, offset:] + decay[:, offset:].exp() * values[:, :-offset],
                ),
                dim=1,
            )
            decay = torch.cat((decay[:, :offset], decay[:, offset:] + decay[:, :-offset]), dim=1)
            offset *= 2
        states = values + decay.exp() * h[:, None]
        h = states[:, -1]
        outputs.append((states * c[:, sl, None, :]).sum(-1) + d * u[:, sl])
    return torch.cat(outputs, dim=1)


def compiled_selective_scan(**kwargs):
    """Create an optionally configured torch.compile version of the scan."""
    return torch.compile(selective_scan, **kwargs)
