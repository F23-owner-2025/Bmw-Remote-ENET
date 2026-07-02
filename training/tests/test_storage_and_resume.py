import json

import pytest

from trainkit.resume import checkpoint_integrity, find_resume_checkpoint
from trainkit.storage import (
    InsufficientDiskError,
    disk_status,
    list_checkpoints,
    prune_checkpoints,
    require_free_gb,
)


def make_ckpt(root, step, complete=True, corrupt_state=False):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    if corrupt_state:
        (d / "trainer_state.json").write_text('{"global_step": ')  # truncated
    else:
        (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    if complete:
        (d / "optimizer.pt").write_bytes(b"x" * 64)
        (d / "adapter_model.safetensors").write_bytes(b"x" * 64)
    return d


# --------------------------- storage ---------------------------------------

def test_disk_status_reports_positive():
    s = disk_status(".")
    assert s.total_gb > 0 and s.free_gb >= 0


def test_require_free_gb_passes_for_tiny_need(tmp_path):
    require_free_gb(tmp_path, 0.001, "test")


def test_require_free_gb_raises_for_absurd_need(tmp_path):
    with pytest.raises(InsufficientDiskError, match="pre-merge test"):
        require_free_gb(tmp_path, 10_000_000, "pre-merge test")


def test_list_checkpoints_sorted_and_filtered(tmp_path):
    make_ckpt(tmp_path, 300)
    make_ckpt(tmp_path, 100)
    make_ckpt(tmp_path, 200)
    (tmp_path / "not-a-checkpoint").mkdir()
    (tmp_path / "checkpoint-abc").mkdir()
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == ["checkpoint-100", "checkpoint-200", "checkpoint-300"]


def test_prune_keeps_newest(tmp_path):
    for s in (100, 200, 300, 400):
        make_ckpt(tmp_path, s)
    deleted = prune_checkpoints(tmp_path, keep=2)
    assert [p.name for p in deleted] == ["checkpoint-100", "checkpoint-200"]
    assert [p.name for p in list_checkpoints(tmp_path)] == \
        ["checkpoint-300", "checkpoint-400"]


def test_prune_noop_when_under_limit(tmp_path):
    make_ckpt(tmp_path, 100)
    assert prune_checkpoints(tmp_path, keep=3) == []


# --------------------------- resume ----------------------------------------

def test_integrity_complete(tmp_path):
    ok, problems = checkpoint_integrity(make_ckpt(tmp_path, 100))
    assert ok, problems


def test_integrity_missing_optimizer(tmp_path):
    d = make_ckpt(tmp_path, 100, complete=False)
    ok, problems = checkpoint_integrity(d)
    assert not ok
    assert any("optimizer" in p for p in problems)


def test_integrity_corrupt_trainer_state(tmp_path):
    d = make_ckpt(tmp_path, 100, corrupt_state=True)
    ok, problems = checkpoint_integrity(d)
    assert not ok
    assert any("corrupt" in p for p in problems)


def test_auto_resume_picks_newest_complete(tmp_path, capsys):
    make_ckpt(tmp_path, 100)
    make_ckpt(tmp_path, 200)
    make_ckpt(tmp_path, 300, complete=False)  # crashed mid-save
    ckpt = find_resume_checkpoint(tmp_path, "auto")
    assert ckpt is not None and ckpt.name == "checkpoint-200"
    assert "skipping incomplete checkpoint checkpoint-300" in capsys.readouterr().out


def test_auto_resume_fresh_start_when_nothing_usable(tmp_path):
    make_ckpt(tmp_path, 100, complete=False)
    assert find_resume_checkpoint(tmp_path, "auto") is None
    assert find_resume_checkpoint(tmp_path / "nonexistent", "auto") is None


def test_never_policy(tmp_path):
    make_ckpt(tmp_path, 100)
    assert find_resume_checkpoint(tmp_path, "never") is None


def test_explicit_path_validates(tmp_path):
    good = make_ckpt(tmp_path, 100)
    assert find_resume_checkpoint(tmp_path, str(good)) == good
    bad = make_ckpt(tmp_path, 200, complete=False)
    with pytest.raises(RuntimeError, match="integrity"):
        find_resume_checkpoint(tmp_path, str(bad))
