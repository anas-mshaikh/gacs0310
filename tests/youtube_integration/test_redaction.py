from src.youtube_integration.redaction import redact_mapping, redact_value


def test_redact_value_masks_middle():
    assert redact_value("abcd1234wxyz") == "abcd...wxyz<redacted>"


def test_redact_mapping_redacts_sensitive_keys():
    payload = {
        "client_id": "safe-client-id",
        "client_secret": "very-secret-value",
        "refresh_token": "refresh-token-value",
        "nested": {
            "access_token": "access-token-value",
            "normal": "visible",
        },
    }

    redacted = redact_mapping(payload)

    assert redacted["client_id"] == "safe-client-id"
    assert redacted["client_secret"] != "very-secret-value"
    assert redacted["refresh_token"] != "refresh-token-value"
    assert redacted["nested"]["access_token"] != "access-token-value"
    assert redacted["nested"]["normal"] == "visible"
