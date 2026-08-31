import pytest
from domains.pcb.preprocessing.annotation import parse_deeppcb_annotation

def test_parse_deeppcb_annotation_valid_comma():
    """
    Test parsing a documented DeepPCB annotation line (comma separated).
    """
    line = "100,150,200,250,3"
    class_id, x_c, y_c, w, h = parse_deeppcb_annotation(line, img_width=640, img_height=640)

    assert class_id == 2  # Type 3 becomes class 2
    assert x_c == 150 / 640.0
    assert y_c == 200 / 640.0
    assert w == 100 / 640.0
    assert h == 100 / 640.0

def test_parse_deeppcb_annotation_valid_space():
    """
    Test parsing an actual DeepPCB annotation line (space separated).
    """
    line = "466 441 493 470 3"
    class_id, x_c, y_c, w, h = parse_deeppcb_annotation(line, img_width=640, img_height=640)

    assert class_id == 2  # Type 3 becomes class 2
    assert x_c == (466 + 493) / 2.0 / 640.0
    assert y_c == (441 + 470) / 2.0 / 640.0
    assert w == (493 - 466) / 640.0
    assert h == (470 - 441) / 640.0

def test_parse_deeppcb_annotation_invalid():
    """
    Test that invalid formats raise appropriate errors.
    """
    with pytest.raises(ValueError):
        parse_deeppcb_annotation("100 150 200 250")  # Missing type

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
