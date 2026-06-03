from harm_detection.utils.text import build_model_text, normalize_text, parse_harm_labels, risk_band, token_count


def test_normalize_text_and_build_model_text() -> None:
    assert normalize_text("  hello   world  ") == "hello world"
    assert normalize_text("0") == ""
    assert build_model_text(" title ", "0", " transcript ") == "title\ntranscript"


def test_parse_harm_labels_filters_invalid_tokens() -> None:
    assert parse_harm_labels("IH, HH, 0, junk, PH") == ["IH", "HH", "PH"]
    assert parse_harm_labels("No agreement") == []


def test_risk_band_and_token_count() -> None:
    assert risk_band(0.10) == "low"
    assert risk_band(0.35) == "medium"
    assert risk_band(0.65) == "medium"
    assert risk_band(0.90) == "high"
    assert token_count("one two three") == 3
