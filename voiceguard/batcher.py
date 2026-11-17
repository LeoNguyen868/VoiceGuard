"""Dynamic batching of windows across concurrent requests and calls.

Windows are queued; the worker takes everything that arrives within max_wait_ms of the first
window, up to max_batch windows, and runs them as one batch. A single worker thread runs the
model, so requests never compete for the device and the event loop stays responsive.
max_requests=1 turns batching across requests off (each request runs alone), which is the
reference point of the load test.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class WindowBatcher:
    def __init__(self, infer, max_batch=16, max_wait_ms=10.0, max_requests=None):
        self.infer, self.max_batch, self.max_wait = infer, max_batch, max_wait_ms / 1000
        self.max_requests = max_requests
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.loop, self.queue, self.task = None, None, None
        self.stats = dict(batches=0, windows=0)

    async def submit(self, clips):
        """clips: list of equal-length window arrays -> N x 2 logits (numpy)."""
        loop = asyncio.get_running_loop()
        if self.task is None or self.loop is not loop:  # first call, or a new event loop (tests)
            self.loop, self.queue = loop, asyncio.Queue()
            self.task = loop.create_task(self._run())
        fut = loop.create_future()
        await self.queue.put((clips, fut))
        return await fut

    async def _collect(self):
        loop = asyncio.get_running_loop()
        items = [await self.queue.get()]
        n, deadline = len(items[0][0]), loop.time() + self.max_wait
        while n < self.max_batch and (self.max_requests is None or len(items) < self.max_requests):
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                if loop.time() >= deadline:
                    break
                await asyncio.sleep(0.001)
                continue
            items.append(item)
            n += len(item[0])
        return items

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            items = await self._collect()
            clips = [c for cs, _ in items for c in cs]
            try:
                logits = await loop.run_in_executor(self.pool, self._infer, clips)
            except Exception as e:  # noqa: BLE001 - forwarded to every waiting request
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            i = 0
            for cs, fut in items:
                if not fut.done():
                    fut.set_result(logits[i:i + len(cs)])
                i += len(cs)
            self.stats["batches"] += 1
            self.stats["windows"] += len(clips)

    def _infer(self, clips):
        return np.concatenate([self.infer(clips[i:i + self.max_batch])
                               for i in range(0, len(clips), self.max_batch)])
