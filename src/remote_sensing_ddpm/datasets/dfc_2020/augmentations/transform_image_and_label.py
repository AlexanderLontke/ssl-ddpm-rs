from typing import Dict

import torch
from torch import nn


class TransformImageAndLabel(nn.Module):
    def __init__(self, image_key, label_key, transform_op: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_key = image_key
        self.label_key = label_key
        self.transform_op = transform_op

    def forward(self, batch: Dict[str, torch.Tensor]):
        keys = [self.image_key, self.label_key]
        assert all(x in batch.keys() for x in keys)
        img_and_label = [batch[k] for k in keys]
        img_and_label = [torch.Tensor(x) for x in img_and_label]
        for i in range(len(img_and_label)):
            if len(img_and_label[i].shape) == 2:
                img_and_label[i] = img_and_label[i].unsqueeze(dim=0)
        original_shapes = [x.shape for x in img_and_label]
        assert all(original_shapes[0][1] == s[1] for s in original_shapes)
        img_and_label = torch.concatenate(tensors=img_and_label, dim=0)
        img_and_label = self.transform_op(img_and_label)
        img_and_label = torch.split(img_and_label, split_size_or_sections=[s[0] for s in original_shapes], dim=0)

        for k, v in zip(keys, img_and_label):
            batch[k] = v
        return batch
