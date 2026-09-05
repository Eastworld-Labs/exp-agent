from system2_agent.model import OpenAICompatibleModel


def test_model_loads_nearest_dotenv_without_overriding_exported_values(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "SYSTEM2_BASE_URL=http://dotenv.test/v1\n"
        "SYSTEM2_API_KEY=dotenv-key\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SYSTEM2_BASE_URL", raising=False)
    monkeypatch.delenv("SYSTEM2_API_KEY", raising=False)
    model = OpenAICompatibleModel.from_env("local-model")
    assert model.base_url == "http://dotenv.test/v1"
    assert model.api_key == "dotenv-key"

    monkeypatch.setenv("SYSTEM2_API_KEY", "exported-key")
    model = OpenAICompatibleModel.from_env("local-model")
    assert model.api_key == "exported-key"


def test_known_provider_uses_its_provider_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setenv("SYSTEM2_API_KEY", "custom-key")
    model = OpenAICompatibleModel.from_env("openai/vision-tool-model")
    assert model.base_url == "https://api.openai.com/v1"
    assert model.api_key == "provider-key"
    assert model.model == "vision-tool-model"
