"""
sampler.py - without-replacement data sampler with a resumable position (needed for continual pretraining).

Every source's token stream is cut into non-overlapping 1024-token windows (+1 token for the target). For each
source and epoch a seeded permutation of the windows is made and split between the ranks (rank r takes
perm[r::world]), so within an epoch no window is repeated and the 2 GPUs never see the same window.
Sources are picked per sequence by the mix weights. The position (epoch + pointer per source, source-choice RNG)
is stored in every checkpoint, so a resumed or continued run carries on with UNSEEN windows instead of re-sampling.
"""
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # shared model code
import copy
import queue
import threading

import numpy as np
import torch


class EpochSampler:
    def __init__(self, data, mb, seed, rank, world, block=1024, state=None):
        self.data, self.mb, self.seed, self.rank, self.world, self.block = data, mb, seed, rank, world, block
        self.sources = data.sources
        self.nwin = {s: (data.bin[s].size - 1) // block for s in self.sources}
        self.epoch = {s: 0 for s in self.sources}
        self.ptr = {s: 0 for s in self.sources}
        self.rng = np.random.default_rng([seed, rank, 777])
        self._shard = {}
        if state is not None:
            self.load(state)

    def _get_shard(self, s):
        key = (s, self.epoch[s])
        if key not in self._shard:
            self._shard = {k: v for k, v in self._shard.items() if k[0] != s}
            perm = np.random.default_rng([self.seed, self.sources.index(s), self.epoch[s]]).permutation(self.nwin[s])
            self._shard[key] = perm[self.rank::self.world]
        return self._shard[key]

    def next_batch(self):
        picks = self.rng.choice(len(self.sources), size=self.mb, p=self.data.weights)
        out = np.empty((self.mb, self.block + 1), dtype=np.int64)
        for r, k in enumerate(picks):
            s = self.sources[k]
            sh = self._get_shard(s)
            w = int(sh[self.ptr[s]])
            out[r] = self.data.bin[s][w * self.block: w * self.block + self.block + 1]
            self.ptr[s] += 1
            if self.ptr[s] >= len(sh):
                self.epoch[s] += 1
                self.ptr[s] = 0
        return torch.from_numpy(out)

    def snapshot(self):
        return copy.deepcopy(dict(epoch=self.epoch, ptr=self.ptr, rng=self.rng.bit_generator.state))

    def load(self, st):
        self.epoch, self.ptr = dict(st["epoch"]), dict(st["ptr"])
        for s in self.sources:
            self.epoch.setdefault(s, 0)
            self.ptr.setdefault(s, 0)
        self._shard = {}

    def epochs_from(self, snap):
        """Fractional passes over each source for a snapshot (this rank's view; ranks are symmetric)."""
        return {s: round(snap["epoch"][s] + snap["ptr"][s] / max(1, self.nwin[s] // self.world), 3) for s in self.sources}


class Prefetch(threading.Thread):
    """Background thread: builds pinned (mb, 1025) batches and remembers the sampler position after each one."""

    def __init__(self, sampler, local=0, n_ahead=6):
        super().__init__(daemon=True)
        self.s, self.local = sampler, local
        self.q = queue.Queue(maxsize=n_ahead)
        self.pin = torch.cuda.is_available()
        self.last_snap = sampler.snapshot()
        self.start()

    def run(self):
        if self.pin:
            torch.cuda.set_device(self.local)
        while True:
            w = self.s.next_batch()
            self.q.put((w.pin_memory() if self.pin else w, self.s.snapshot()))

    def get(self, device):
        w, snap = self.q.get()
        self.last_snap = snap
        return w.to(device, non_blocking=True)
