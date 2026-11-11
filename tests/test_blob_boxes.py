from blob_boxes.fit import fit_boxes
import numpy as np
import pytest

def test_blob_boxes():
    mask = np.zeros((20, 20))
    mask[2:5, 3:9] = 1
    mask[7:10, 2:3] = 1

    results = fit_boxes(mask)

    assert len(results) == 2

    assert results[0].box.cx == pytest.approx(5.5, abs=0.01)
    assert results[0].box.cy == pytest.approx(3.0, abs=0.01)
    assert results[0].box.width == pytest.approx(6.0, abs=0.01)
    assert results[0].box.height == pytest.approx(3.0, abs=0.01)
    assert results[0].iou == pytest.approx(1.0, abs=0.01)
    assert results[0].box.angle == pytest.approx(0.0, abs=0.01)

    assert results[1].box.cx == pytest.approx(2.0, abs=0.01)
    assert results[1].box.cy == pytest.approx(8.0, abs=0.01)
    assert results[1].box.width == pytest.approx(3.0, abs=0.01)
    assert results[1].box.height == pytest.approx(1.0, abs=0.01)
    assert results[1].iou == pytest.approx(1.0, abs=0.01)
    assert results[1].box.angle == pytest.approx(-90.0, abs=0.01)