import torch

from comfy_kitchen.prefetch_ring import record_region, set_record_region


def test_record_region_forwards_tensors_in_order():
    seen = []
    first = torch.empty(3)
    second = torch.empty(5)
    set_record_region(seen.append)
    try:
        record_region(first)
        record_region(second)
    finally:
        set_record_region(None)

    assert len(seen) == 2
    assert seen[0] is first
    assert seen[1] is second


def test_record_region_is_noop_without_recorder():
    set_record_region(None)
    record_region(torch.empty(1))
