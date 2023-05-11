from typing import Iterator, List

# FFCV
from ffcv.loader import Loader


class FFCVLoaderWrapper(Iterator):
    def __init__(self, mapping: List[str], *args, **kwargs):
        self.original_dataloader = Loader(*args, **kwargs)
        self.mapping = mapping

    def __iter__(self):
        self.original_dataloader_it = self.original_dataloader.__iter__()
        self.len = self.original_dataloader_it.__len__()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        next_batch = next(self.original_dataloader_it)
        assert len(self.mapping) == len(next_batch)
        result_batch = {}
        for i, name in enumerate(self.mapping):
            result_batch[name] = next_batch[i]
        return result_batch
