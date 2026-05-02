from safemail.interpretability import extract_suspicious_indicators, highlight_text


def test_extract_suspicious_indicators_detects_urls_and_terms():
    text = "Please verify your account immediately at https://safe-mail.test/login"
    indicators = extract_suspicious_indicators(text)
    assert len(indicators["urls"]) == 1
    assert "verify" in indicators["urgency_terms"]
    assert "immediately" in indicators["urgency_terms"]


def test_highlight_text_marks_matches():
    text = "Urgent action required. Click now."
    html = highlight_text(text)
    assert "<mark>" in html
