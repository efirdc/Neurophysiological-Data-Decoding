from typing import Any, Sequence, Mapping

import torch

def is_sequence(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def to_device(elem: Any, device: torch.device):
    if is_sequence(elem):
        return [to_device(e, device) for e in elem]
    if isinstance(elem, Mapping):
        return {k: to_device(v, device) for k, v in elem.items()}
    if isinstance(elem, torch.Tensor):
        return elem.to(device)
    return elem
