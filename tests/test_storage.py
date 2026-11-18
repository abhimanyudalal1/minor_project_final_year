import shutil

import pytest

from burgers_research.common import load_config
from burgers_research.storage import check_storage, storage_budget


def test_full_study_fits_existing_capacity(tmp_path, monkeypatch):
    cfg = load_config("configs/study.yaml")
    monkeypatch.setattr("burgers_research.storage.shutil.disk_usage",
                        lambda _: shutil._ntuple_diskusage(100*10**9, 28*10**9, 72*10**9))
    budget = check_storage(cfg, tmp_path)
    assert budget["reference_bytes"] == 2750 * 201 * 8192 * 8
    assert budget["derived_bytes"] == 2750 * 201 * (256+512+1024) * 4
    assert 40e9 < budget["required_free_bytes"] < 72e9
    monkeypatch.setattr("burgers_research.storage.shutil.disk_usage",
                        lambda _: shutil._ntuple_diskusage(100*10**9, 99*10**9, 10**9))
    with pytest.raises(RuntimeError, match="estimated remaining"):
        check_storage(cfg, tmp_path)


def test_resume_credits_existing_artifacts(cfg, tmp_path):
    before = storage_budget(cfg, tmp_path)
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "completed-shard.h5").write_bytes(b"x" * 4096)
    after = storage_budget(cfg, tmp_path)
    assert after["required_free_bytes"] == before["required_free_bytes"] - 4096
