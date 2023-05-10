from typing import Iterator, List

from torch.utils.data import DataLoader
from ffcv.loader import Loader


class FFCVLoaderWrapper(Iterator):
    def __init__(self, mapping: List[str], *args, **kwargs):
        self.original_dataloader = Loader(*args, **kwargs)
        self.len = self.original_dataloader.__len__()
        self.original_dataloader_it = self.original_dataloader.__iter__()
        self.mapping = mapping

    def __len__(self):
        return self.len

    def __next__(self):
        next_batch = next(self.original_dataloader_it)
        assert len(self.mapping) == len(next_batch)
        result_batch = {}
        for i, name in enumerate(self.mapping):
            result_batch[name] = next_batch[i]
        return result_batch
