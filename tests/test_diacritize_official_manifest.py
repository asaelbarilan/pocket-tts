"""The diacritize pass and the vocalized-row schema.

Everything here runs with the fake diacritizer, so it needs no GPU and no model download. Its
output is not real niqqud.
"""

import itertools
import json
import unicodedata
from pathlib import Path

import pytest

from hebrew_training import align_official_manifest as aligner
from hebrew_training import diacritize_official_manifest as diac
from hebrew_training import diacritizer, manifest_files
from hebrew_training.diacritizer import fake_diacritizer
from hebrew_training.nikud import strip_marks, validate_nikud, vocalize_word_lists, word_chunks

SENTENCES = [
    "אדוני היושב ראש אני מבקש להעלות את הנושא לסדר היום",
    "חברי הכנסת הנכבדים מדובר בהחלטה משמעותית עבור אזרחי המדינה",
    "הוועדה תתכנס ביום שלישי הקרוב בשעה עשר בבוקר",
    "תודה רבה אני מסיים כאן את דבריי",
    "שלום עולם",
]


def make_row(index: int, text: str | None = None, **extra) -> dict:
    text = text or SENTENCES[index % len(SENTENCES)]
    words = text.split()
    return {
        "path": f"/audio/rec-{index // 3}.wav",
        "start": float(index * 20),
        "duration": float(len(words)),
        "transcript": text,
        "speaker": f"speaker-{index // 3}",
        "words": [
            {"word": word, "start": float(n), "end": n + 0.9} for n, word in enumerate(words)
        ],
        **extra,
    }


def write_manifest(folder: Path, name: str, rows: list[dict]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(folder: Path, *extra: str) -> None:
    diac.main([str(folder), "--diacritizer", "fake", *extra])


# --- the schema -------------------------------------------------------------------------


def test_strip_is_exact_for_the_fake_and_handles_unicode_forms():
    words = SENTENCES[0].split()
    marked = fake_diacritizer(words)
    assert all(m != w for m, w in zip(marked, words))
    validate_nikud(words, marked)
    # shin + shin dot written as a presentation form, a dagesh after the vowel, a cantillation
    # mark: NFC reorders and decomposes, and every one is a mark.
    assert strip_marks("\ufb2a") == "ש"
    assert strip_marks("בּ\u05b8") == strip_marks("ב\u05b8\u05bc") == "ב"
    assert strip_marks("ו\u0591") == "ו"
    # punctuation survives, including the maqaf: it is not a mark.
    assert strip_marks("שָׁלוֹם־עוֹלָם, 12 ok.") == "שלום־עולם, 12 ok."


@pytest.mark.parametrize(
    "words, nikud, message",
    [
        (["שלום"], [], "0 entries for 1"),
        (["שלום"], ["שָׁלוֹם", "x"], "2 entries for 1"),
        (["שלום"], ["שָׁלוֹם עוֹלָם"], "not a single word"),
        (["שלום"], ["עוֹלָם"], "does not strip back"),
        (["שלום"], [""], "not a single word"),
        (["שלום"], ["שלום "], "not a single word"),
    ],
)
def test_validate_nikud_rejects_what_does_not_strip_back(words, nikud, message):
    with pytest.raises(ValueError, match=message):
        validate_nikud(words, nikud)


def test_word_chunks_respect_the_diacritizer_line_limit():
    words = ["אבגדהוז"] * 100
    chunks = word_chunks(words, max_chars=100)
    assert chunks[0][0] == 0 and chunks[-1][1] == 100
    assert all(a[1] == b[0] for a, b in itertools.pairwise(chunks))
    assert all(len(" ".join(words[a:b])) <= 100 for a, b in chunks)
    assert word_chunks([]) == []
    assert word_chunks(["x" * 500], max_chars=100) == [(0, 1)]


def test_vocalize_word_lists_reports_rows_that_do_not_strip_back():
    rows = [["שלום", "עולם"], ["אבג"], ["דהו", "זחט"]]

    def predict(lines):
        out = fake_diacritizer(lines)
        out[1] = "ת" + out[1][1:]  # changes a letter
        out[2] = out[2].split()[0]  # drops a word
        return out

    results = vocalize_word_lists(rows, predict, batch_size=3, max_chars=400)
    validate_nikud(rows[0], results[0])
    assert results[1] == "content_changed"
    assert results[2] == "word_count_changed"


def test_a_long_row_is_vocalized_in_pieces_and_put_back_together():
    words = [w for s in SENTENCES for w in s.split()] * 6
    calls = []

    def predict(lines):
        calls.append(lines)
        return fake_diacritizer(lines)

    (result,) = vocalize_word_lists([words], predict, batch_size=2, max_chars=120)
    validate_nikud(words, result)
    assert sum(map(len, calls)) > 3 and all(len(line) <= 120 for c in calls for line in c)


# --- the pass ---------------------------------------------------------------------------


def test_untouched_rows_are_copied_byte_for_byte_and_picked_rows_only_gain_nikud(tmp_path):
    rows = [make_row(i) for i in range(60)]
    rows[5]["alt_words"] = {"norms": [{"from": 1, "to": 1, "replace": ["14"]}]}
    source = write_manifest(tmp_path, "train.jsonl", rows)
    write_manifest(tmp_path, "valid.jsonl", rows[:5])
    run(tmp_path, "--rate", "0.3")

    before = source.read_bytes().splitlines(keepends=True)
    after = (tmp_path / "train_diac.jsonl").read_bytes().splitlines(keepends=True)
    assert len(before) == len(after) == 60
    vocalized = 0
    for original, new in zip(before, after):
        if original == new:
            continue
        vocalized += 1
        old, row = json.loads(original), json.loads(new)
        nikud = row.pop("nikud")
        assert row == old  # nothing else changed, including words, times and alt_words
        validate_nikud([w["word"] for w in old["words"]], nikud)
    assert 5 <= vocalized <= 35  # 0.3 of 60, give or take the hash
    assert source.read_bytes() == b"".join(before)  # the source is never modified
    summary = json.loads((tmp_path / "train_diac.summary.json").read_text())
    assert summary["stats"]["diacritized"] == vocalized and summary["synthetic"] is True
    assert not (tmp_path / "train_diac.progress.json").exists()


def test_the_same_rows_are_picked_on_every_run_and_the_seed_changes_them(tmp_path):
    rows = [make_row(i) for i in range(80)]
    for name in ("a", "b", "c"):
        write_manifest(tmp_path / name, "train.jsonl", rows)
    run(tmp_path / "a", "--splits", "train")
    run(tmp_path / "b", "--splits", "train", "--batch-rows", "7")
    run(tmp_path / "c", "--splits", "train", "--seed", "1")
    a, b, c = (read(tmp_path / n / "train_diac.jsonl") for n in "abc")

    def picked(rows):
        return [i for i, row in enumerate(rows) if "nikud" in row]

    assert picked(a) == picked(b) != picked(c)


def test_rate_zero_changes_nothing_and_rate_one_vocalizes_every_eligible_row(tmp_path):
    rows = [make_row(i) for i in range(20)]
    rows.append(make_row(20, "123 456"))  # no Hebrew letter: never eligible
    rows.append(make_row(21, "שָׁלוֹם עוֹלָם"))  # already has marks: never eligible
    write_manifest(tmp_path / "zero", "train.jsonl", rows)
    write_manifest(tmp_path / "one", "train.jsonl", rows)
    run(tmp_path / "zero", "--splits", "train", "--rate", "0")
    run(tmp_path / "one", "--splits", "train", "--rate", "1")
    assert (tmp_path / "zero" / "train_diac.jsonl").read_bytes() == (
        tmp_path / "zero" / "train.jsonl"
    ).read_bytes()
    one = read(tmp_path / "one" / "train_diac.jsonl")
    assert ["nikud" in row for row in one] == [True] * 20 + [False, False]
    stats = json.loads((tmp_path / "one" / "train_diac.summary.json").read_text())["stats"]
    assert stats["ineligible_no_hebrew"] == 1 and stats["ineligible_has_marks"] == 1


def test_there_is_no_length_rule_a_very_long_row_is_vocalized_like_any_other(tmp_path):
    long_text = " ".join(SENTENCES * 8)  # ~ 300 words, far past any text window
    write_manifest(tmp_path, "train.jsonl", [make_row(0, long_text), make_row(1, "שלום")])
    run(tmp_path, "--splits", "train", "--rate", "1")
    rows = read(tmp_path / "train_diac.jsonl")
    assert all("nikud" in row for row in rows)
    validate_nikud([w["word"] for w in rows[0]["words"]], rows[0]["nikud"])
    assert len(rows[0]["nikud"]) > 250


def test_validation_gets_every_eligible_row_by_default_and_train_the_rate(tmp_path):
    rows = [make_row(i) for i in range(40)]
    write_manifest(tmp_path, "train.jsonl", rows)
    write_manifest(tmp_path, "valid.jsonl", rows)
    run(tmp_path)
    assert all("nikud" in row for row in read(tmp_path / "valid_diac.jsonl"))
    train = read(tmp_path / "train_diac.jsonl")
    assert 0 < sum("nikud" in row for row in train) < 40


def test_input_that_is_not_a_built_manifest_is_refused_before_anything_runs(tmp_path):
    write_manifest(tmp_path, "train.jsonl", [make_row(0), {"path": "x", "start": 0}])
    with pytest.raises(SystemExit, match="row 2 has no numeric `duration`"):
        run(tmp_path, "--splits", "train")
    assert not (tmp_path / "train_diac.jsonl").exists()
    write_manifest(tmp_path, "train.jsonl", [make_row(0, nikud=["x"] * 11)])
    with pytest.raises(SystemExit, match="nikud"):
        run(tmp_path, "--splits", "train")
    write_manifest(tmp_path, "train.jsonl", [make_row(0, words=[])])
    with pytest.raises(SystemExit, match="no `words` list"):
        run(tmp_path, "--splits", "train")
    (tmp_path / "train.jsonl").write_text("not json\n")
    with pytest.raises(SystemExit, match="not a JSON-lines"):
        run(tmp_path, "--splits", "train")


def test_a_file_that_was_already_diacritized_is_not_diacritized_again(tmp_path):
    write_manifest(tmp_path, "train.jsonl", [make_row(0)])
    run(tmp_path, "--splits", "train", "--rate", "1")
    (tmp_path / "train.jsonl").write_bytes((tmp_path / "train_diac.jsonl").read_bytes())
    with pytest.raises(SystemExit, match="already has `nikud`"):
        run(tmp_path, "--splits", "train", "--overwrite")


def test_an_existing_output_is_kept_unless_overwrite(tmp_path, capsys):
    write_manifest(tmp_path, "train.jsonl", [make_row(0)])
    run(tmp_path, "--splits", "train", "--rate", "1")
    first = (tmp_path / "train_diac.jsonl").read_bytes()
    run(tmp_path, "--splits", "train", "--rate", "0")
    assert (tmp_path / "train_diac.jsonl").read_bytes() == first
    assert "already exists" in capsys.readouterr().out
    run(tmp_path, "--splits", "train", "--rate", "0", "--overwrite")
    assert (tmp_path / "train_diac.jsonl").read_bytes() == (tmp_path / "train.jsonl").read_bytes()


def test_an_interrupted_run_resumes_to_the_same_bytes(tmp_path):
    rows = [make_row(i) for i in range(50)]
    write_manifest(tmp_path / "whole", "train.jsonl", rows)
    write_manifest(tmp_path / "cut", "train.jsonl", rows)
    run(tmp_path / "whole", "--splits", "train", "--batch-rows", "8")
    run(tmp_path / "cut", "--splits", "train", "--batch-rows", "8", "--limit", "21")
    cut = tmp_path / "cut"
    assert len((cut / "train_diac.jsonl").read_bytes().splitlines()) == 21
    progress = json.loads((cut / "train_diac.progress.json").read_text())
    assert progress["input_lines"] == 21
    # a differently configured resume is refused rather than mixed in
    with pytest.raises(SystemExit, match="different configuration"):
        run(cut, "--splits", "train", "--seed", "9")
    run(cut, "--splits", "train", "--batch-rows", "8")
    assert (cut / "train_diac.jsonl").read_bytes() == (
        tmp_path / "whole" / "train_diac.jsonl"
    ).read_bytes()
    assert not (cut / "train_diac.progress.json").exists()


# --- the three stages can run in either order -------------------------------------------


def aligned_like(row: dict) -> dict:
    """What the aligner does to a row: new times, rebuilt transcript, everything else kept."""
    timed = [
        {"word": item["word"], "start": 0.5 * n, "end": 0.5 * n + 0.4}
        for n, item in enumerate(row["words"])
    ]
    return aligner.update_timings(row, 0.0, timed, "viter")


def test_aligning_keeps_nikud_and_alt_words_and_refuses_to_change_the_word_count():
    row = make_row(
        0,
        nikud=fake_diacritizer(SENTENCES[0].split()),
        alt_words={"norms": [{"from": 1, "to": 2, "replace": ["14"]}]},
    )
    out = aligned_like(row)
    for key in ("nikud", "alt_words", "speaker", "path"):
        assert out[key] == row[key]
    assert [w["word"] for w in out["words"]] == [w["word"] for w in row["words"]]
    assert out["transcript"] == SENTENCES[0] and out["alignment_backend"] == "viter"
    short = [
        {"word": w["word"], "start": float(n), "end": n + 0.5}
        for n, w in enumerate(row["words"][:-1])
    ]
    with pytest.raises(ValueError, match="words for"):
        aligner.update_timings(row, 0.0, short, "viter")


def test_diacritizing_then_aligning_equals_aligning_then_diacritizing(tmp_path):
    rows = [
        make_row(i, alt_words={"norms": [{"from": 0, "to": 0, "replace": ["7"]}]})
        for i in range(25)
    ]
    write_manifest(tmp_path / "dna", "train.jsonl", rows)
    run(tmp_path / "dna", "--splits", "train", "--rate", "0.5")
    diacritized_then_aligned = [
        aligned_like(r) for r in read(tmp_path / "dna" / "train_diac.jsonl")
    ]

    aligned_first = [aligned_like(r) for r in rows]
    write_manifest(tmp_path / "and", "train.jsonl", aligned_first)
    # the aligned rows carry alignment keys the diacritizer must leave alone; it does not
    # forbid them, only an existing `nikud`
    run(tmp_path / "and", "--splits", "train", "--rate", "0.5")
    aligned_then_diacritized = read(tmp_path / "and" / "train_diac.jsonl")

    assert any("nikud" in r for r in aligned_then_diacritized)
    assert diacritized_then_aligned == aligned_then_diacritized


def test_stage_file_names_are_the_same_in_the_aligner_and_the_shared_module():
    assert aligner.BUILT == manifest_files.BUILT
    assert aligner.DIACRITIZED == manifest_files.DIACRITIZED
    assert aligner.ALIGNED == manifest_files.ALIGNED
    assert aligner.VERIFY_ROWS == manifest_files.VERIFY_ROWS


def test_every_stage_name_tells_which_stage_wrote_it():
    names = [manifest_files.BUILT, manifest_files.DIACRITIZED, manifest_files.ALIGNED]
    assert [n.format(split="train") for n in names] == [
        "train.jsonl",
        "train_diac.jsonl",
        "train_aligned.jsonl",
    ]


def test_nfc_form_is_what_gets_stored(tmp_path):
    # The fake never emits a non-NFC string, but the diacritizer can; vocalize_word_lists
    # normalizes before it splits, so what is stored is always NFC.
    def predict(lines):
        return [unicodedata.normalize("NFD", line) for line in fake_diacritizer(lines)]

    (result,) = vocalize_word_lists([["שלום"]], predict)
    assert result == [unicodedata.normalize("NFC", result[0])]


# --- GPUs --------------------------------------------------------------------------------


@pytest.fixture
def four_gpus(monkeypatch):
    gpus = [
        diacritizer.Gpu(0, "NVIDIA A40", 44.4),
        diacritizer.Gpu(1, "NVIDIA A40", 44.4),
        diacritizer.Gpu(2, "NVIDIA L4", 22.5),
        diacritizer.Gpu(3, "NVIDIA A40", 44.4),
    ]
    monkeypatch.setattr(diacritizer, "detect_gpus", lambda: gpus)
    return gpus


def test_every_detected_gpu_is_used_by_default_and_reported_by_type(four_gpus):
    devices, text = diacritizer.resolve_devices("all", "dicta")
    assert devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert "found 4 GPU(s)" in text and "2: NVIDIA L4 (22.5 GiB)" in text
    assert "Using 4: 0,1,2,3" in text and "NVIDIA A40 / NVIDIA L4" in text


def test_gpus_can_be_limited_and_reordered_by_index(four_gpus):
    devices, text = diacritizer.resolve_devices("3, 1", "dicta")
    assert devices == ["cuda:3", "cuda:1"] and "Using 2: 3,1" in text
    assert diacritizer.resolve_devices("2", "dicta")[0] == ["cuda:2"]


@pytest.mark.parametrize("bad", ["4", "0,9", "1,1", "a,b", ""])
def test_gpus_that_do_not_exist_are_refused_with_the_list_of_those_that_do(four_gpus, bad):
    with pytest.raises(SystemExit, match="--gpus"):
        diacritizer.resolve_devices(bad, "dicta")


def test_cpu_and_mps_are_explicit_and_no_gpu_is_only_an_error_for_the_real_model(monkeypatch):
    monkeypatch.setattr(diacritizer, "detect_gpus", list)
    assert diacritizer.resolve_devices("cpu", "dicta")[0] == ["cpu"]
    assert diacritizer.resolve_devices("mps", "dicta")[0] == ["mps"]
    assert diacritizer.resolve_devices("all", "fake")[0] == ["cpu"]
    with pytest.raises(SystemExit, match="no CUDA GPU found"):
        diacritizer.resolve_devices("all", "dicta")


def test_several_gpus_give_the_same_bytes_as_one_device(four_gpus, tmp_path, capsys):
    rows = [make_row(i) for i in range(60)]
    for name in ("one", "many"):
        write_manifest(tmp_path / name, "train.jsonl", rows)
    # tiny batches, so the batches really are spread over the workers and must be reordered
    run(tmp_path / "one", "--splits", "train", "--gpus", "cpu")
    run(
        tmp_path / "many",
        "--splits",
        "train",
        "--gpus",
        "1,3",
        "--batch-size",
        "2",
        "--batch-rows",
        "10",
    )
    assert (tmp_path / "many" / "train_diac.jsonl").read_bytes() == (
        tmp_path / "one" / "train_diac.jsonl"
    ).read_bytes()
    assert "Using 2: 1,3" in capsys.readouterr().out


def test_the_default_batch_of_rows_grows_with_the_gpus(four_gpus, tmp_path, monkeypatch):
    seen = []
    real = diac.run_split

    def spy(args, *a, **k):
        seen.append(args.batch_rows)
        return real(args, *a, **k)

    monkeypatch.setattr(diac, "run_split", spy)
    write_manifest(tmp_path, "train.jsonl", [make_row(i) for i in range(6)])
    run(tmp_path, "--splits", "train", "--gpus", "0,1,2,3")
    run(tmp_path, "--splits", "train", "--gpus", "cpu", "--overwrite")
    assert seen == [2 * 32 * 4, 128]


def test_a_worker_that_cannot_load_its_model_fails_the_run_with_its_reason(four_gpus, tmp_path):
    write_manifest(tmp_path, "train.jsonl", [make_row(0)])
    with pytest.raises(RuntimeError, match="diacritizer worker .* failed"):
        diac.main(
            [
                str(tmp_path),
                "--splits",
                "train",
                "--model",
                str(tmp_path / "no-such-model"),
                "--gpus",
                "0,1",
            ]
        )
    with pytest.raises(SystemExit, match="needs --model"):
        diac.main([str(tmp_path), "--splits", "train", "--gpus", "0,1"])


def test_a_worker_sees_one_gpu_as_cuda_0(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert diacritizer._physical_gpu(3) == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4, 6,7")
    assert diacritizer._physical_gpu(1) == "6"
