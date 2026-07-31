# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Combined DRAM-core prefetch + consuming 1D matmul.

``ttnn.experimental.queue_tensor_prefetcher_request`` (fills a DRAM-sender
GlobalCircularBuffer over NOC, off the command queue) and the ``ttnn.linear``
that drains that GCB are always issued as a pair, against the *same* GCB and the
*same* gather_in0 program config. As two separate calls the caller has to (a)
hand both the same ``global_cb``, (b) hand both the same ``program_config``, and
(c) pass a prefetch ``block_count`` that matches what the matmul expects -- three
couplings nothing enforces.

``prefetch_and_linear`` issues the pair from one call site so they cannot drift:
it derives ``block_count`` from ``global_cb``, queues the request, then runs the
consuming ``ttnn.linear`` with the same GCB and program config. The matmul does
``wait_front(ring_size)`` per layer, so the only correct ``block_count`` is the
GCB's receiver count -- which holds for every weight layout (WIDTH_SHARDED and
receiver-contiguous alike), so the caller never specifies it. The weight↔config
geometry was already validated when the GCB was built (and ``ttnn.linear``
re-checks it), so there is nothing left to cross-check here.

This is a host-side composition, not a device-level fusion: the prefetch still
runs on the DRAM-core (DRISC) path off the command queue while the matmul is
dispatched normally. A ``queue_id``/``cq_id`` in ``**linear_kwargs`` still reaches
``ttnn.linear``, but is read out here as well so that it steers both halves: applied
to only the matmul it would leave the prefetch on whatever queue was already current,
and a capture region would then capture the halves apart.
"""

import ttnn


def prefetch_and_linear(
    input_tensor_a,
    weight,
    *,
    global_cb,
    program_config,
    **linear_kwargs,
):
    """Queue a DRAM-core prefetch of ``weight`` into ``global_cb``, then run the
    gather_in0 1D matmul (``ttnn.linear``) that consumes it.

    Batched vs streaming delivery is selected automatically from
    ``program_config.stream_in1`` -- there is no separate argument. When it is set the
    prefetch streams the weight's K-blocks in natural ring order (identity rotation) so
    the matmul can consume them FIFO from a shallow GCB; this works for both
    ROUND_ROBIN_1D and CONTIGUOUS_1D receiver-contiguous weights. When it is unset the
    request is batched (the whole per-receiver slab is delivered before the matmul reads).

    Args:
        input_tensor_a: Activation (in0), width-sharded on the receiver cores.
        weight: DRAM-sharded weight (in1) to prefetch and multiply by. Streaming
            (``stream_in1``) additionally requires a receiver-contiguous weight layout.
        global_cb: DRAM-sender GlobalCircularBuffer shared by the prefetch and
            the matmul. Its receiver count fixes the prefetch ``block_count``.
        program_config: gather_in0 1D mcast matmul program config driving the matmul;
            its ``stream_in1`` flag selects streaming vs batched prefetch delivery.
        **linear_kwargs: Forwarded to ``ttnn.linear`` (e.g. ``memory_config``,
            ``compute_kernel_config``, ``dtype``, ``bias``). A ``queue_id``/``cq_id``
            here steers both halves -- the prefetch is captured against that queue (when
            it is recording a trace) and the matmul dispatches on it -- and defaults to
            the calling thread's current queue, as set by ``ttnn.command_queue(n)``.
            Either spelling is accepted, as for any ttnn operation; ``queue_id`` wins if
            both are given, matching ttnn's own precedence.

    Returns:
        The ``ttnn.linear`` output tensor.
    """
    device = input_tensor_a.device()
    # block_count == ring_size == total GCB receivers: the matmul does
    # wait_front(ring_size) per layer, so this is the only value that balances the
    # page credits, regardless of the weight's shard layout.
    block_count = global_cb.receiver_cores().num_cores()
    # Streaming vs batched is decided entirely by the matmul's program config, so the
    # caller passes no extra argument. With ``stream_in1`` the matmul consumes K-blocks
    # FIFO as they land (letting the GCB hold only a shallow window), so the prefetch
    # must deliver them in natural ring order -- an identity rotation table
    # (``rotation[r] = r``). That table is layout-agnostic: it is identical for
    # ROUND_ROBIN_1D and CONTIGUOUS_1D weights because the kernel slices it by each
    # weight's own global receiver position, so no distribution-strategy argument is
    # needed here. Without ``stream_in1`` the matmul waits for the whole per-receiver
    # slab, so we queue the batched (rotation-free) request.
    if program_config.stream_in1:
        request = (weight, block_count, list(range(block_count)))
    else:
        request = (weight, block_count)
    # Read (not popped) out of the kwargs the way ttnn's own operation wrapper resolves it
    # -- FastOperation.__call__ in ttnn/decorators.py picks on the keyword being present, not
    # on its value, so an explicit queue_id=None wins over a cq_id rather than falling through
    # to it. Only the prefetch half needs it named here; the keyword itself stays in
    # linear_kwargs and reaches ttnn.linear as the caller wrote it.
    cq_id = None
    if "queue_id" in linear_kwargs:
        cq_id = linear_kwargs["queue_id"]
    elif "cq_id" in linear_kwargs:
        cq_id = linear_kwargs["cq_id"]
    ttnn.experimental.queue_tensor_prefetcher_request(
        device,
        [request],
        global_cb=global_cb,
        # Capture against the queue the matmul below dispatches on, so both halves land in the
        # one trace. Left False, a capture region would take the matmul but send the prefetch
        # immediately -- a replay would never refill the GCB and the matmul would hang.
        capture_into_trace=True,
        cq_id=cq_id,
    )
    return ttnn.linear(
        input_tensor_a,
        weight,
        program_config=program_config,
        global_cb=global_cb,
        **linear_kwargs,
    )
