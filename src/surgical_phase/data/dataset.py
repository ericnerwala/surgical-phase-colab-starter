from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class Sample:
    frame_path: str
    phase: int
    video_id: str


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224, augment: bool = False):
        self.samples: List[Sample] = [
            Sample(r.frame_path, int(r.phase), r.video_id) for r in df.itertuples(index=False)
        ]
        t = [transforms.Resize((image_size, image_size))]
        if augment:
            t += [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.RandomHorizontalFlip(0.5)]
        t += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.tf = transforms.Compose(t)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.frame_path).convert("RGB")
        return self.tf(img), torch.tensor(s.phase, dtype=torch.long), s.video_id


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 32, image_size: int = 224, stride: int = 8, augment: bool = False):
        self.df = df.sort_values(["video_id", "frame_id"]).reset_index(drop=True)
        self.seq_len = seq_len
        self.stride = stride
        t = [transforms.Resize((image_size, image_size))]
        if augment:
            t += [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)]
        t += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.tf = transforms.Compose(t)

        self.windows = []
        for vid, sub in self.df.groupby("video_id"):
            idxs = list(sub.index)
            for start in range(0, max(1, len(idxs) - seq_len + 1), stride):
                end = min(start + seq_len, len(idxs))
                chunk = idxs[start:end]
                if len(chunk) < seq_len:
                    chunk = chunk + [chunk[-1]] * (seq_len - len(chunk))
                self.windows.append(chunk)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        indices = self.windows[idx]
        rows = self.df.iloc[indices]
        frames = [self.tf(Image.open(p).convert("RGB")) for p in rows.frame_path.tolist()]
        x = torch.stack(frames, dim=0)  # [T, C, H, W]
        y = torch.tensor(rows.phase.iloc[-1], dtype=torch.long)  # next/current phase target at sequence end
        vid = rows.video_id.iloc[-1]
        return x, y, vid
