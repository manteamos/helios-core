"""
Celery application — broker, result backend, and pool configuration.

Worker pool choice: prefork (default on Linux)
----------------------------------------------
NumPy releases the GIL during most array operations, so threads can run
in parallel on a single CPython interpreter.  We choose prefork anyway for
three reasons specific to this workload:

1. Memory isolation.  Each worker process owns its full address space.
   Multi-GB raster datasets can be memory-mapped independently without
   one worker's I/O fragmenting another's heap.

2. GIL spikes on short strides.  The transposition and FD thermal loops
   operate on 8760-row arrays with strides as small as 8 bytes.  CPython
   re-acquires the GIL at the end of each NumPy "ufunc chunk" (default
   ~1000 elements), causing contention spikes in a threaded pool that
   prefork sidesteps entirely.

3. Fault isolation.  An OOM kill (plausible for a 25-year Monte Carlo run
   on a small instance) hits only the affected worker process; the pool
   supervisor respawns it without disturbing in-flight tasks on peers.
"""

from __future__ import annotations

from celery import Celery

from api.config import settings

celery_app = Celery(
    "helios",
    broker=settings.redis_url,
    backend=settings.result_backend,
    include=["api.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=settings.task_result_ttl_seconds,
    worker_prefetch_multiplier=1,  # one task at a time; prevents head-of-line blocking
    task_acks_late=True,  # ack only after successful completion; no silent drop on OOM
    worker_pool="prefork",
    worker_concurrency=None,  # defaults to os.cpu_count()
    task_track_started=True,
    task_send_sent_event=True,
)
