"""Same cache samples/windows; transport raw BF16 instead of CPU FP32 to reduce IPC/H2D traffic."""
import bisect
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .vlm_frontend_v1 import bits_bf16


class VLMCacheData(Dataset):
    def __init__(self, cache, numeric_cache, part, horizon=10, max_open_episodes=12):
        self.cache = Path(cache)
        complete = json.loads((self.cache/"complete.json").read_text())
        if complete["status"] not in ("COMPLETE_REVIEW_REQUIRED", "PASS"):
            raise ValueError("Full cache required")
        self.index = json.loads((self.cache/part/"index.json").read_text())
        self.ends = [e["end"] for e in self.index]
        if not self.index or self.index[0]["start"] != 0:
            raise ValueError("Invalid episode index")
        for a, b in zip(self.index, self.index[1:]):
            if a["end"] != b["start"]:
                raise ValueError("Noncontiguous index")
        self.numeric = {k: np.load(Path(numeric_cache)/part/(k+".npy"), mmap_mode="r")
                        for k in ("state", "actions", "episode_id", "task_id")}
        assert len(self.numeric["state"]) == self.ends[-1] == complete["parts"][part]["frames"]
        for e in self.index:
            assert np.all(self.numeric["episode_id"][e["start"]:e["end"]] == e["episode_id"])
        self.horizon, self.limit, self.opened = horizon, max_open_episodes, OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def window_indices(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        e = self.index[bisect.bisect_right(self.ends, index)]
        requested = np.arange(index, index+self.horizon)
        return np.minimum(requested, e["end"]-1), requested < e["end"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        e = self.index[bisect.bisect_right(self.ends, index)]
        eid = e["episode_id"]
        if eid not in self.opened:
            self.opened[eid] = np.load(self.cache/e["directory"]/"hidden.npy", mmap_mode="r")
            while len(self.opened) > self.limit:
                self.opened.popitem(last=False)
        self.opened.move_to_end(eid)
        hidden = bits_bf16(self.opened[eid][index-e["start"]])
        target, valid = self.window_indices(index)
        return dict(hidden=hidden, hidden_mask=torch.ones(len(hidden), dtype=torch.bool),
                    state=torch.from_numpy(np.array(self.numeric["state"][index])),
                    actions=torch.from_numpy(np.array(self.numeric["actions"][target, 0])),
                    action_mask=torch.from_numpy(valid), episode_id=eid,
                    frame_index=index-e["start"], task_id=int(self.numeric["task_id"][index]))


def collate_cached(samples):
    b, longest, width = len(samples), max(len(s["hidden"]) for s in samples), samples[0]["hidden"].shape[-1]
    hidden = torch.zeros(b, longest, width, dtype=torch.bfloat16)
    mask = torch.zeros(b, longest, dtype=torch.bool)
    for i, s in enumerate(samples):
        n = len(s["hidden"])
        hidden[i, :n] = s["hidden"]
        mask[i, :n] = True
    result = dict(hidden=hidden, hidden_mask=mask)
    for key in ("state", "actions", "action_mask"):
        result[key] = torch.stack([s[key] for s in samples])
    for key in ("episode_id", "frame_index", "task_id"):
        result[key] = torch.tensor([s[key] for s in samples], dtype=torch.long)
    return result
