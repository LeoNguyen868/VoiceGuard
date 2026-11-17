"""CUDA Graphs for fixed-length windows.

Every window has the same shape (4 s = 64,000 samples), so the whole forward pass can be
captured once per batch size and replayed. Replay removes the per-kernel launch overhead,
which is what makes the classifier take 11-12 ms on the GPU regardless of the input length.
A batch is padded with zeros up to the nearest captured size; in eval mode every sample is
processed independently, so the padding does not change the other outputs.
"""
import torch


class GraphRunner:
    def __init__(self, model, device="cuda", n_samples=64000, batch_sizes=(1, 2, 4, 8, 16), fp16=True):
        if not str(device).startswith("cuda"):
            raise ValueError("CUDA Graphs need a CUDA device")
        self.model, self.device, self.n_samples, self.fp16 = model.eval(), device, n_samples, fp16
        self.sizes = sorted(batch_sizes)
        self.graphs = {}
        pool = torch.cuda.graph_pool_handle()
        for b in reversed(self.sizes):  # largest first, so smaller graphs reuse its memory pool
            self.graphs[b] = self._capture(b, pool)

    def _forward(self, x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16, cache_enabled=False):
            return self.model(x)

    @torch.no_grad()
    def _capture(self, b, pool):
        static_in = torch.zeros(b, self.n_samples, device=self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # warm-up outside the graph (cuDNN/cuBLAS autotuning, allocations)
                self._forward(static_in)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            static_out = self._forward(static_in)
        return graph, static_in, static_out

    @torch.no_grad()
    def __call__(self, x):
        """x: N x n_samples tensor (any device) -> N x 2 float32 logits on the GPU."""
        x = x.to(self.device, non_blocking=True)
        outs, i = [], 0
        while i < len(x):
            n = len(x) - i
            b = next((s for s in self.sizes if s >= n), self.sizes[-1])
            take = min(n, b)
            graph, static_in, static_out = self.graphs[b]
            static_in[:take].copy_(x[i:i + take])
            if take < b:
                static_in[take:].zero_()
            graph.replay()
            outs.append(static_out[:take].float().clone())
            i += take
        return torch.cat(outs)
