import pytest

from vaillant_rag.config import Settings, load_settings


def test_defaults_are_valid():
    Settings().validate()


def test_yaml_file_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("chunk_size_chars: 500\nchunk_overlap_chars: 50\n", encoding="utf-8")
    settings = load_settings(str(config))
    assert settings.chunk_size_chars == 500
    assert settings.chunk_overlap_chars == 50


def test_env_overrides_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("top_k: 10\n", encoding="utf-8")
    monkeypatch.setenv("TOP_K", "7")
    settings = load_settings(str(config))
    assert settings.top_k == 7


def test_bool_parsing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_HYBRID_SEARCH", "false")
    assert load_settings().use_hybrid_search is False
    monkeypatch.setenv("USE_HYBRID_SEARCH", "1")
    assert load_settings().use_hybrid_search is True


def test_overlap_geq_size_rejected():
    with pytest.raises(ValueError, match="chunk_overlap_chars"):
        Settings(chunk_size_chars=100, chunk_overlap_chars=100).validate()


def test_top_n_greater_than_top_k_rejected():
    with pytest.raises(ValueError, match="top_n_contexts"):
        Settings(top_k=3, top_n_contexts=5).validate()


def test_unknown_yaml_key_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("no_such_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_settings(str(config))


def test_invalid_int_value_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOP_K", "not-a-number")
    with pytest.raises(ValueError, match="TOP_K|top_k"):
        load_settings()


def test_inline_system_prompt():
    settings = Settings(system_prompt_path="You are concise.")
    assert settings.system_prompt == "You are concise."
