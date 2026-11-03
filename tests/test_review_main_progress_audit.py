import pytest

from tools.review_main_progress_audit_20261024 import correct_unrenderable, output_bytes, summarize_cells


def escaped(raw):
    return b"".join(b"\x01"+bytes([b]) for b in raw)


def test_output_byte_count_removes_codec_tags_and_eos():
    body="影评".encode()
    wire=escaped(body)+b"\x00\x00"
    assert output_bytes(dict(visible_bytes_hex=wire.hex(),total_visible_bytes=len(wire)))==body


def test_unrenderable_tail_keeps_bytes_and_bits_and_separates_text_delivery():
    raw="影评".encode()+b"\xe7\xb8"
    wire=escaped(raw)
    result=dict(visible_bytes_hex=wire.hex(),total_visible_bytes=len(wire),framing_status="invalid_utf8")
    row=dict(case_id="x",model="qwen_instruct",top_k=0,variant="las_imec",method="las",
        text=None,bytes=0,strict_verified_bits=16,exact=True,sender_status="ok",receiver_status="ok",
        nll_total=10.,evaluator_tokens=5,body_tokens=5)
    corrected=correct_unrenderable(row,dict(result=result))
    assert row["bytes"]==0
    assert corrected["bytes"]==8 and corrected["strict_verified_bits"]==16
    assert corrected["incomplete_utf8_suffix_bytes"]==2
    cell=summarize_cells([corrected])[0]
    assert cell["byte_delivery_bpb"]==pytest.approx(2.)
    assert cell["valid_text_delivery_bpb"]==0
    assert cell["exact_receivers"]==1 and cell["scored"]==1
    assert cell["unrenderable_outputs"]==1
