import pytest
from domains.pcb.preprocessing.annotation import parse_deeppcb_annotation

def test_parse_deeppcb_annotation_valid():
    """
    Test parsing a typical DeepPCB annotation line:
    x1,y1,x2,y2,type -> class_id, x_center, y_center, width, height
    type is 1-6, class_id should be 0-5.
    """
    # Example: x1=100, y1=150, x2=200, y2=250, type=3 (mouse_bite)
    # class_id = 3 - 1 = 2
    # width = 100/640 = 0.15625, height = 100/640 = 0.15625
    # x_center = 150/640 = 0.234375, y_center = 200/640 = 0.3125
    line = "100,150,200,250,3"
    class_id, x_c, y_c, w, h = parse_deeppcb_annotation(line, img_width=640, img_height=640)

    assert class_id == 2  # Type 3 becomes class 2
    assert x_c == 150 / 640.0
    assert y_c == 200 / 640.0
    assert w == 100 / 640.0
    assert h == 100 / 640.0

def test_parse_deeppcb_annotation_invalid():
    """
    Test that invalid formats raise appropriate errors.
    """
    with pytest.raises(ValueError):
        parse_deeppcb_annotation("100 150 200 250 3")  # Space separated instead of comma

    with pytest.raises(ValueError):
        parse_deeppcb_annotation("100,150,200,250")  # Missing type

def test_parse_deeppcb_annotation_edge_cases():
    """
    Test edge cases for coordinates (e.g., 0 and max dimensions).
    """
    line = "0,0,640,640,1"
    class_id, x_c, y_c, w, h = parse_deeppcb_annotation(line, img_width=640, img_height=640)

    assert class_id == 0
    assert x_c == 0.5
    assert y_c == 0.5
    assert w == 1.0
    assert h == 1.0
