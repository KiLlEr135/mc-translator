"""Tests for StringEstimator._count_generic_json_file -- the pre-flight
"Подсчёт строк" count for generic JSON files, which must match what
GenericJsonProcessor will actually process (same lenient parsing/encoding)."""
from mc_translator.processors.estimator import StringEstimator
from mc_translator.runtime.state import JobState


def make_estimator():
    return StringEstimator(JobState())


LANG = {"file": "ru_ru", "api": "ru", "regex": r"[А-Яа-яЁё]"}


def test_counts_strict_json_normally(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"tooltip": "Right click to open"}', encoding="utf-8")
    assert make_estimator()._count_generic_json_file(str(path), LANG, "force") == 1


def test_counts_jsonc_with_comments_and_trailing_commas(tmp_path):
    """Regression test: GenericJsonProcessor reads source files via
    load_lenient_json (tolerates // and /* */ comments and trailing
    commas -- common in mod config JSON), but the estimator used bare
    json.loads, which raises on these and silently returns 0 -- undercounting
    a file the processor would translate normally."""
    path = tmp_path / "config.json"
    path.write_text(
        '{\n  // a comment\n  "tooltip": "Right click to open",\n}',
        encoding="utf-8",
    )
    assert make_estimator()._count_generic_json_file(str(path), LANG, "force") == 1


def test_counts_json_with_utf8_bom(tmp_path):
    """Regression test: GenericJsonProcessor opens source files with
    utf-8-sig (strips a leading BOM), but the estimator used plain utf-8,
    which leaves the BOM attached to the first key and breaks JSON parsing
    entirely -- silently undercounting (returns 0) a file that parses fine
    for the actual processor."""
    path = tmp_path / "config.json"
    path.write_bytes('{"tooltip": "Right click to open"}'.encode("utf-8-sig"))
    assert make_estimator()._count_generic_json_file(str(path), LANG, "force") == 1
