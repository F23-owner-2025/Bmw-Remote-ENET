import textwrap

import pytest

from trainkit.config import ConfigError, RunConfig, load_config, validate_config


def write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_defaults_are_valid():
    assert validate_config(RunConfig()) == []


def test_minimal_yaml_loads(tmp_path):
    cfg = load_config(write(tmp_path, """
        run_name: t
        stage: sft
    """))
    assert cfg.run_name == "t"
    assert cfg.lora.r == 64
    assert cfg.model.max_seq_len == 16384


def test_overrides_apply(tmp_path):
    cfg = load_config(write(tmp_path, """
        stage: sft
        lora: {r: 16, alpha: 32}
        optim: {learning_rate: 2.0e-4}
    """))
    assert cfg.lora.r == 16
    assert cfg.optim.learning_rate == pytest.approx(2e-4)


def test_all_errors_collected(tmp_path):
    with pytest.raises(ConfigError) as ei:
        load_config(write(tmp_path, """
            stage: nonsense
            lora: {r: -1}
            optim: {learning_rate: 0.5}
            bogus_section: {}
        """))
    msg = str(ei.value)
    assert "stage" in msg
    assert "lora.r" in msg
    assert "learning_rate" in msg
    assert "bogus_section" in msg


def test_unknown_section_key_rejected(tmp_path):
    with pytest.raises(ConfigError) as ei:
        load_config(write(tmp_path, """
            stage: sft
            optim: {learning_rte: 1.0e-4}
        """))
    assert "learning_rte" in str(ei.value)


def test_missing_file():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/cfg.yaml")


def test_orpo_constraints(tmp_path):
    with pytest.raises(ConfigError) as ei:
        load_config(write(tmp_path, """
            stage: orpo
            orpo: {max_length: 1024, max_prompt_length: 2048}
        """))
    assert "max_prompt_length" in str(ei.value)


def test_shipped_configs_load():
    from pathlib import Path

    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    names = {p.name for p in cfg_dir.glob("*.yaml")}
    assert {"sft_plan_a.yaml", "sft_qlora_fallback.yaml", "orpo_plan_a.yaml"} <= names
    for p in cfg_dir.glob("*.yaml"):
        cfg = load_config(p)  # must not raise
        assert cfg.stage in ("sft", "orpo")
