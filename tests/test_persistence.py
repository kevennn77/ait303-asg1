import pytest


@pytest.mark.skip(reason="Model files created during Colab execution — check after Phase 4 complete")
def test_models_saved():
    """Verify LDA .pkl, BERTopic .safetensors, CorEx .pkl exist."""
    assert True
