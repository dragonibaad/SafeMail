from safemail.data import LABEL_MAP, clean_text, preprocess_dataset


def test_clean_text_removes_url_and_noise():
    raw = "URGENT! Verify at https://abc.com now!!!"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "!" not in cleaned
    assert "urgent" in cleaned


def test_preprocess_dataset_applies_label_map():
    import pandas as pd

    df = pd.DataFrame(
        {
            "Email Text": ["hello", "verify account now"],
            "Email Type": ["Safe Email", "Phishing Email"],
        }
    )
    processed = preprocess_dataset(df)
    assert set(processed["label_num"].tolist()) == {LABEL_MAP["Safe Email"], LABEL_MAP["Phishing Email"]}
