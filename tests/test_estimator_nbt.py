"""Regression test for StringEstimator._count_nbt_file -- it used to call
nbtlib.File.load(fileobj) (which raises on nbtlib>=2.0.4, which requires a
path and an explicit `gzipped` flag) and check isinstance() against
tag.CompoundTag/ListTag/StringTag, none of which exist on that nbtlib version
(the real names are tag.Compound/List/String). The isinstance() calls raised
AttributeError with no surrounding try/except, so a modpack with even one NBT
file crashed the whole Analyze/cost-estimate pass."""
import pytest

nbtlib = pytest.importorskip("nbtlib")
from nbtlib import tag  # noqa: E402

from mc_translator.processors.estimator import StringEstimator


def test_count_nbt_file_does_not_raise_and_counts_translatable_strings(tmp_path, job_state):
    path = tmp_path / "structure.nbt"
    nbtlib.File(tag.Compound({
        "sign": tag.String("Hello adventurer"),
        "items": tag.List[tag.Compound]([
            tag.Compound({"name": tag.String("A translatable name")}),
        ]),
        "id": tag.String("minecraft.stone"),  # technical term -- not counted
    })).save(str(path))

    est = StringEstimator(job_state)
    count = est._count_nbt_file(str(path), r"[А-Яа-яЁё]", "force")
    assert count == 2


def test_count_nbt_file_returns_zero_on_unreadable_file(tmp_path, job_state):
    path = tmp_path / "not_really_nbt.nbt"
    path.write_bytes(b"not an nbt file")

    est = StringEstimator(job_state)
    assert est._count_nbt_file(str(path), r"[А-Яа-яЁё]", "force") == 0
