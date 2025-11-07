import numpy as np
import pytest
from ligotools import readligo

def test_dq_channel_to_seglist_simple():
    # I shall do a simple teest on dq_channel_to_seglist boolean array
    channel = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    segments = readligo.dq_channel_to_seglist(channel, fs=1)
    # I should get indices 1-3 and 4-7
    expected = [slice(1, 3), slice(4, 7)]
    
    # Here I shall compare the start and stop for each slice
    for seg, exp in zip(segments, expected):
        assert seg.start == exp.start
        assert seg.stop == exp.stop

def test_segmentlist_init_from_list():

    # Here I will test SegmentList constructor with a list input
    seg_list = [[100, 200], [300, 400]]
    seg_obj = readligo.SegmentList(seg_list)
    
    assert seg_obj.seglist == seg_list
    # To see whethere __getitem__ works
    assert seg_obj[0] == [100, 200]
    # Seeing if __iter__ works
    segs = [s for s in seg_obj]
    assert segs == seg_list
