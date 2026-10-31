"""Dataloader behaviour that silently degrades training when it breaks."""

import json
import random
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import sphn

from hebrew_training import build_official_manifest as builder


def _import_dataloader():
    try:
        import training.dataloader as module
    except ImportError:
        # `pocket_tts` needs scipy's native library, which is broken on some dev Macs. The
        # loader only needs `convert_audio` from it (a sample-rate mismatch), so stub it
        # for the import and leave sys.modules as it was.
        stub = types.ModuleType("pocket_tts.data.audio_utils")
        stub.convert_audio = mock.Mock(side_effect=RuntimeError("pocket_tts is stubbed"))
        with mock.patch.dict(sys.modules, {"pocket_tts.data.audio_utils": stub}):
            import training.dataloader as module
    return module


td = _import_dataloader()
DataLoader, Entry, load_entries = td.DataLoader, td.Entry, td.load_entries

SR = 24000
FIXTURE = Path(__file__).parent / "fixtures" / "valid_diac.jsonl"


def _write_wav(path, seconds=6.0):
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    sphn.write_wav(str(path), (0.2 * np.sin(2 * np.pi * 140 * t)).astype(np.float32), SR)


def _manifest(tmp_path, n=8, words=True, duration=6.0):
    wav = tmp_path / "a.flac"
    _write_wav(wav, duration)
    path = tmp_path / "m.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            entry = {"path": str(wav), "duration": duration, "transcript": "one two three four"}
            if words:
                entry["words"] = [
                    {"word": w, "start": 0.8 * j, "end": 0.8 * j + 0.6}
                    for j, w in enumerate(entry["transcript"].split())
                ]
            f.write(json.dumps(entry) + "\n")
    return str(path)


def _loader(manifest, **kw):
    kw.setdefault("batch_size", 2)
    return DataLoader(
        manifest,
        lambda s: [1] * max(1, len(s) // 3),
        kw.pop("batch_size"),
        SR,
        12.5,
        kw.pop("max_duration_sec", 30.0),
        kw.pop("max_voice_prompt_sec", 3.0),
        0,
        1,
        seed=0,
        shuffle=False,
        **kw,
    )


def test_rank_sharding_partitions_entries_without_overlap(tmp_path):
    m = _manifest(tmp_path, n=8)
    shards = [load_entries(m, rank, 4) for rank in range(4)]
    assert sum(len(s) for s in shards) == 8
    assert all(len(s) == 2 for s in shards)


def test_prompt_respects_the_configured_cap(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path), max_voice_prompt_sec=1.0)))
    assert (batch.num_voice_prompt_frames.float() / 12.5).max().item() <= 1.0 + 1e-6


def test_batches_have_the_requested_size(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path, n=8), batch_size=4)))
    assert batch.audio.shape[0] == 4
    assert len(batch.text_tokens) == 4


def test_target_audio_never_exceeds_max_duration(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path, duration=30.0), max_duration_sec=5.0)))
    assert batch.audio.shape[-1] <= int(5.0 * SR) + 1


def test_unaligned_manifest_still_yields_batches(tmp_path):
    """Manifests without word alignments are a documented input: the loader
    falls back to a random window as the prompt instead of hanging."""
    loader = _loader(_manifest(tmp_path, n=8, words=False))
    batch = next(iter(loader))
    assert batch.audio.shape[0] == 2


def test_entry_start_offsets_into_a_shared_file(tmp_path):
    """Two utterances can share one audio file: each entry reads its own
    window, at `start`, out of the shared file."""
    low_hz, high_hz = 220, 880
    t = np.linspace(0, 10.0, int(10.0 * SR), endpoint=False)
    wav = np.where(
        t < 5.0, 0.2 * np.sin(2 * np.pi * low_hz * t), 0.2 * np.sin(2 * np.pi * high_hz * t)
    )
    audio_path = tmp_path / "shared.flac"
    sphn.write_wav(str(audio_path), wav.astype(np.float32), SR)

    manifest = tmp_path / "m.jsonl"
    with open(manifest, "w") as f:
        f.write(json.dumps({"path": str(audio_path), "duration": 5.0, "transcript": "low"}) + "\n")
        f.write(
            json.dumps(
                {"path": str(audio_path), "duration": 5.0, "transcript": "high", "start": 5.0}
            )
            + "\n"
        )

    loader = _loader(str(manifest), batch_size=2)
    low_entry, high_entry = loader.get_entry(0), loader.get_entry(1)
    assert low_entry.start == 0.0
    assert high_entry.start == 5.0

    low_wav, *_ = loader._sample(low_entry)
    high_wav, *_ = loader._sample(high_entry)

    def dominant_freq(x):
        spectrum = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1 / SR)
        return freqs[np.argmax(spectrum)]

    assert abs(dominant_freq(low_wav) - low_hz) < 2
    assert abs(dominant_freq(high_wav) - high_hz) < 2


# ------------------------------------------------------------------ digit / niqqud text


@pytest.fixture
def silent_audio(monkeypatch):
    """No file is read: the text tests only look at what is handed to the tokenizer."""
    monkeypatch.setattr(
        td, "_load_window", lambda path, start, dur, sr: np.zeros(max(1, int(dur * sr)), np.float32)
    )


class _PickCut:
    """Stands in for `loader.rng` so the cut is the one a test names (the real draw is shared
    between pool threads, and the text tests are about everything else)."""

    def __init__(self, word_index):
        self.word_index = word_index

    def choice(self, cuts):
        return next(c for c in cuts if c[1] == self.word_index)

    def uniform(self, a, b):
        return a


def _fixture_rows():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _plain(row):
    return {k: v for k, v in row.items() if k not in ("alt_words", "nikud")}


def _text_loader(tmp_path, rows, texts, **kw):
    manifest = tmp_path / "text.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    kw.setdefault("fixed_draws", False)

    def tokenize(text):
        texts.append(text)
        return [1]

    return DataLoader(
        str(manifest), tokenize, 1, SR, 12.5, 30.0, 5.0, 0, 1, seed=kw.pop("seed", 0),
        shuffle=False, **kw,
    )  # fmt: skip


def _words(row):
    return [w["word"] for w in row["words"]]


def test_get_entry_carries_alt_words_and_nikud(tmp_path):
    row = _fixture_rows()[0]
    entry = _text_loader(tmp_path, [row], []).get_entry(0)
    assert entry.alt_words == row["alt_words"]
    assert entry.nikud == row["nikud"]
    assert entry.text_seed is not None


def test_old_rows_are_trained_exactly_as_before_and_draw_nothing(tmp_path, silent_audio):
    old = [_plain(r) for r in _fixture_rows()]
    texts, texts_off = [], []
    loader = _text_loader(tmp_path, old, texts)
    off = _text_loader(
        tmp_path, old, texts_off,
        alt_words_sample_prob=0, alt_word_swap_prob=0, nikud_word_prob=0,
    )  # fmt: skip
    initial = loader.rng.getstate()
    for index, row in enumerate(old):
        entry = loader.get_entry(index)
        assert entry.text_seed is None
        for pick in (3, 6):
            loader.rng = _PickCut(pick)
            loader._sample(entry)
            off.rng = _PickCut(pick)
            off._sample(off.get_entry(index))
            assert texts[-1] == " ".join(_words(row)[pick:])
    assert texts == texts_off
    loader.rng = random.Random(0)
    assert loader.rng.getstate() == initial
    assert loader._text_rng.getstate() == random.Random("text:0").getstate()


def test_old_rows_leave_the_seeded_cut_stream_bit_identical(tmp_path, silent_audio):
    """Same seed, same rows without the new keys: the cuts (hence the texts) are exactly the
    sequence the loader produced before, which is the join of `words` at each drawn cut."""
    old = [_plain(r) for r in _fixture_rows()] * 5
    texts = []
    loader = _text_loader(tmp_path, old, texts)
    reference = random.Random(0)  # what the pre-change loader consumed: one choice per sample
    for index, row in enumerate(old):
        loader._sample(loader.get_entry(index))
        cuts = []
        for i in range(1, len(row["words"])):
            prev, cur = row["words"][i - 1], row["words"][i]
            cut = 0.5 * (prev["end"] + cur["start"])
            if cut >= 1.0 and row["duration"] - cut >= 1.0:
                cuts.append((cut, i))
        eligible = [(c, i) for c, i in cuts if row["words"][i - 1]["end"] < 5.0]
        _, i = reference.choice(eligible or cuts[:1])
        assert texts[-1] == " ".join(_words(row)[i:])
    assert loader.rng.getstate() == reference.getstate()


def test_text_augmentation_never_touches_the_cut_stream(tmp_path):
    rows = _fixture_rows() * 3
    loader = _text_loader(tmp_path, rows, [])
    initial = loader.rng.getstate()
    for index in range(len(rows)):
        loader.get_entry(index)
    assert loader.rng.getstate() == initial


def test_validation_draws_are_the_same_on_every_evaluation(tmp_path, silent_audio):
    rows = _fixture_rows()
    runs = []
    for _ in range(3):
        texts = []
        loader = _text_loader(tmp_path, rows, texts, fixed_draws=True)
        for index in range(len(rows)):
            loader._sample(loader.get_entry(index))  # the cut is the row's own, not stubbed
        runs.append(texts)
    assert runs[0] == runs[1] == runs[2]
    # ... and the same however many rows came before: a row's draws depend on the row only.
    texts = []
    loader = _text_loader(tmp_path, rows[::-1], texts, fixed_draws=True)
    for index in range(len(rows)):
        loader._sample(loader.get_entry(index))
    assert texts == runs[0][::-1]
    # a different seed is a different (but again fixed) draw
    other = _text_loader(tmp_path, rows, [], fixed_draws=True, seed=1)
    assert (
        other.get_entry(0).text_seed
        != _text_loader(tmp_path, rows, [], fixed_draws=True).get_entry(0).text_seed
    )


def test_training_draws_differ_between_visits_but_replay_with_the_seed(tmp_path, silent_audio):
    rows = _fixture_rows()[:1] * 40

    def run():
        texts = []
        loader = _text_loader(tmp_path, rows, texts)
        for index in range(len(rows)):
            loader.rng = _PickCut(4)
            loader._sample(loader.get_entry(index))
        return texts

    first = run()
    assert len(set(first)) > 5, "a row should vary from visit to visit in training"
    assert run() == first


def test_cut_text_gets_nikud_and_digits_after_the_cut_only(tmp_path, silent_audio):
    row = _fixture_rows()[0]
    words, nikud, spans = _words(row), row["nikud"], row["alt_words"]["norms"]
    cut = 8  # span 5..6 is before the cut; spans 10..12, 14..15 and 20..21 survive
    texts = []
    loader = _text_loader(
        tmp_path, [row], texts,
        alt_words_sample_prob=1, alt_word_swap_prob=1, nikud_word_prob=1,
    )  # fmt: skip
    loader.rng = _PickCut(cut)
    loader._sample(loader.get_entry(0))
    assert texts == [" ".join(builder.swap_alternates(nikud, spans, cut))]
    assert "מ-65" not in texts[0] and "משישים" not in texts[0]

    for kwargs, expected in (
        ({"nikud_word_prob": 1}, " ".join(nikud[cut:])),
        ({"alt_words_sample_prob": 1, "alt_word_swap_prob": 1},
         " ".join(builder.swap_alternates(words, spans, cut))),
        ({}, " ".join(words[cut:])),
    ):  # fmt: skip
        texts.clear()
        plain = {"alt_words_sample_prob": 0, "alt_word_swap_prob": 0, "nikud_word_prob": 0}
        loader = _text_loader(tmp_path, [row], texts, **{**plain, **kwargs})
        loader.rng = _PickCut(cut)
        loader._sample(loader.get_entry(0))
        assert texts == [expected]


def test_a_cut_inside_a_digit_span_leaves_it_spoken(tmp_path, silent_audio):
    row = _fixture_rows()[0]
    cut = 11  # inside the span 10..12 ("אלפיים עשרים וחמש" for 2025)
    texts = []
    loader = _text_loader(
        tmp_path, [row], texts, alt_words_sample_prob=1, alt_word_swap_prob=1, nikud_word_prob=0
    )
    loader.rng = _PickCut(cut)
    loader._sample(loader.get_entry(0))
    spoken = _words(row)[cut : cut + 2]
    assert texts[0].split()[:2] == spoken
    assert "2025" not in texts[0].split()


def test_no_cut_fallback_uses_the_words_for_rows_with_variants(tmp_path, silent_audio):
    short = {
        "path": "x", "duration": 1.5, "transcript": "TRANSCRIPT NOT WORDS",
        "words": [{"word": w, "start": 0.4 * i, "end": 0.4 * i + 0.3}
                  for i, w in enumerate(["לפני", "ארבעה", "עשר", "ימים"])],
        "alt_words": {"norms": [{"from": 1, "to": 2, "replace": ["14"]}]},
        "nikud": ["לִפְנֵי", "אַרְבָּעָה", "עָשָׂר", "יָמִים"],
    }  # fmt: skip
    old = _plain(short)
    for row, kwargs, expected in (
        (short, {"nikud_word_prob": 1, "alt_words_sample_prob": 0}, " ".join(short["nikud"])),
        (short, {"nikud_word_prob": 0, "alt_words_sample_prob": 1, "alt_word_swap_prob": 1},
         "לפני 14 ימים"),
        (short, {"nikud_word_prob": 1, "alt_words_sample_prob": 1, "alt_word_swap_prob": 1},
         "לִפְנֵי 14 יָמִים"),
        # no variants for this row: the transcript, exactly as before
        (old, {"nikud_word_prob": 1, "alt_words_sample_prob": 1}, "TRANSCRIPT NOT WORDS"),
        # both features off: the transcript too
        (short, {"nikud_word_prob": 0, "alt_words_sample_prob": 0}, "TRANSCRIPT NOT WORDS"),
    ):  # fmt: skip
        texts = []
        loader = _text_loader(tmp_path, [row], texts, **kwargs)
        loader._sample(loader.get_entry(0))
        assert texts == [expected], kwargs


def test_probabilities_outside_zero_one_are_rejected_by_the_loader(tmp_path):
    for name in ("alt_words_sample_prob", "alt_word_swap_prob", "nikud_word_prob"):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match=name):
                _text_loader(tmp_path, _fixture_rows(), [], **{name: bad})


def test_loader_defaults_are_the_documented_ones(tmp_path):
    aug = _text_loader(tmp_path, _fixture_rows(), []).text_aug
    assert (aug.alt_words_sample_prob, aug.alt_word_swap_prob, aug.nikud_word_prob) == (
        0.30,
        0.60,
        0.10,
    )


# ------------------------------------------------------------------ validation cuts, prefetch


def _many_rows(count=40):
    """Distinct rows (distinct `start`, so distinct seeds) with plenty of eligible cuts."""
    rows = []
    for n in range(count):
        row = dict(_fixture_rows()[n % 2])
        row["start"] = 100.0 * n
        rows.append(row)
    return rows


def _windows_read_in(monkeypatch, loader, order, workers):
    """Every (start, duration) the loader asked the audio layer for, sampling `order` on a pool."""

    calls = []
    monkeypatch.setattr(
        td,
        "_load_window",
        lambda path, start, dur, sr: (
            calls.append((round(start, 6), round(dur, 6))),
            np.zeros(max(1, int(dur * sr)), np.float32),
        )[1],
    )
    entries = [loader.get_entry(i) for i in order]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(loader._sample_or_none, entries))
    return sorted(calls)


def test_validation_cuts_do_not_depend_on_thread_scheduling(tmp_path, monkeypatch):
    rows = _many_rows()
    count = len(rows)
    orders = [list(range(count)), list(reversed(range(count))), random.Random(3).sample(
        range(count), count
    )]  # fmt: skip
    seen = []
    for order in orders:
        for workers in (1, 8, 16):
            loader = _text_loader(tmp_path, rows, [], fixed_draws=True)
            initial = loader.rng.getstate()
            seen.append(_windows_read_in(monkeypatch, loader, order, workers))
            assert loader.rng.getstate() == initial, "validation must not use the shared stream"
    assert all(s == seen[0] for s in seen), "a row's cut changed with the order it ran in"
    # the cuts really vary from row to row: this is not one constant cut
    assert len({dur for start, dur in seen[0] if start % 100 == 0 and dur < 6}) > 5


def test_training_cuts_still_come_from_the_shared_stream(tmp_path, silent_audio):
    loader = _text_loader(tmp_path, _many_rows(5), [])
    assert all(loader.get_entry(i).cut_seed is None for i in range(5))
    initial = loader.rng.getstate()
    loader._sample(loader.get_entry(0))
    assert loader.rng.getstate() != initial


def test_validation_fallback_prompt_window_is_fixed_per_row(tmp_path, monkeypatch):
    row = {"path": "x", "duration": 30.0, "transcript": "no alignment here"}  # no `words`
    starts = []
    for _ in range(2):
        calls = []
        monkeypatch.setattr(
            td, "_load_window",
            lambda p, s, d, sr, calls=calls: (calls.append(s), np.zeros(int(d * sr), np.float32))[1],
        )  # fmt: skip
        loader = _text_loader(tmp_path, [row], [], fixed_draws=True)
        loader._sample(loader.get_entry(0))
        starts.append(calls[1])
    assert starts[0] == starts[1]
    assert starts[0] > 0, "a window drawn from the row's own generator, not always the start"


def _within(seconds, fn):
    """Run `fn` in a thread; fail instead of hanging if it does not finish."""

    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"still running after {seconds}s: the consumer is hung"
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_prefetch_delivers_items_then_re_raises_the_workers_error():
    class Boom(RuntimeError):
        pass

    def batches():
        yield "first"
        yield "second"
        raise Boom("the loader failed")

    got = []

    def run():
        try:
            for item in td._prefetch(batches()):
                got.append(item)  # noqa: PERF402 -- items before the error must be kept
        except Boom as exc:
            return exc

    error = _within(10, run)
    assert got == ["first", "second"]
    assert isinstance(error, Boom) and "the loader failed" in str(error)


def test_prefetch_still_ends_normally():
    assert _within(10, lambda: list(td._prefetch(iter(["a", "b", "c"])))) == ["a", "b", "c"]


def test_an_unreadable_manifest_raises_instead_of_hanging(tmp_path, monkeypatch):
    def unreadable(*args, **kwargs):
        raise OSError("no such audio")

    monkeypatch.setattr(td, "_load_window", unreadable)
    loader = _text_loader(tmp_path, _fixture_rows(), [])
    with pytest.raises(ValueError, match="no readable samples"):
        _within(20, lambda: next(iter(loader)))


# ------------------------------------------------------------------ stopping the loader


def _wait_until(condition, seconds=5.0):

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def _endless(closed):
    """A loader stand-in: yields forever, and records that it was closed."""
    try:
        n = 0
        while True:
            yield n
            n += 1
    finally:
        closed.set()


def test_closing_prefetch_stops_a_worker_blocked_on_a_full_queue():

    closed = threading.Event()
    before = threading.active_count()
    batches = td._prefetch(_endless(closed), depth=2)
    assert next(batches) == 0
    assert _wait_until(lambda: threading.active_count() > before)  # the worker is up ...

    time.sleep(0.3)  # ... and has filled the queue, so it is blocked in put()

    batches.close()

    assert closed.wait(5), "the loader's own generator must be closed by the worker"
    assert _wait_until(lambda: threading.active_count() <= before), "worker thread leaked"


def test_closing_works_as_a_context_manager_like_validate_uses_it():

    closed = threading.Event()
    before = threading.active_count()
    with closing(td._prefetch(_endless(closed))) as batches:
        got = [next(batches) for _ in range(3)]  # stop early, like num_valid_batches
    assert got == [0, 1, 2]
    assert closed.wait(5)
    assert _wait_until(lambda: threading.active_count() <= before)


def test_closing_prefetch_waits_for_worker_cleanup():
    closed = threading.Event()

    def batches():
        try:
            while True:
                yield 0
        finally:
            time.sleep(0.1)  # stand in for native audio / I/O pool cleanup
            closed.set()

    prefetched = td._prefetch(batches(), depth=1)
    next(prefetched)
    _within(5, prefetched.close)
    assert closed.is_set(), "close must finish cleanup, not just request it"


def test_a_prefetch_that_never_started_leaves_no_thread():

    closed = threading.Event()
    before = threading.active_count()
    td._prefetch(_endless(closed)).close()
    assert threading.active_count() == before


@pytest.fixture
def recorded_pools(monkeypatch):
    """The loader's thread pools, with whether each was shut down."""

    pools = []

    class Pool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.was_shut_down = False
            pools.append(self)

        def shutdown(self, *args, **kwargs):
            self.was_shut_down = True
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(td, "ThreadPoolExecutor", Pool)
    return pools


def test_closing_the_loader_shuts_its_io_pool_down(tmp_path, silent_audio, recorded_pools):
    loader = _text_loader(tmp_path, _fixture_rows(), [])
    batches = loader._batches()
    next(batches)
    assert [p.was_shut_down for p in recorded_pools] == [False]
    batches.close()
    assert [p.was_shut_down for p in recorded_pools] == [True]


def test_closing_the_prefetched_loader_shuts_its_io_pool_down(
    tmp_path, silent_audio, recorded_pools
):
    loader = _text_loader(tmp_path, _fixture_rows(), [])
    batches = iter(loader)
    next(batches)
    batches.close()
    assert recorded_pools and recorded_pools[0].was_shut_down
    assert all(not thread.is_alive() for pool in recorded_pools for thread in pool._threads)


def test_a_loader_that_fails_its_checks_shuts_its_io_pool_down(tmp_path, recorded_pools):
    loader = _text_loader(tmp_path, _fixture_rows(), [])
    loader.batch_size = 99  # more than the rows it has
    with pytest.raises(ValueError, match="a batch can never be filled"):
        next(loader._batches())
    assert [p.was_shut_down for p in recorded_pools] == [True]


# ------------------------------------------------------------------ silence after the last word


def _target_seconds(tmp_path, monkeypatch, tail_gap, **row_overrides):
    """Seconds of target audio asked for, from a row whose last word ends with the row (as an
    aligned row does) and whose only cut is at 2.1 s."""
    row = {
        "path": "x.wav",
        "start": 100.0,
        "duration": 5.0,
        "transcript": "a b",
        "words": [{"word": "a", "start": 0.0, "end": 2.0}, {"word": "b", "start": 2.2, "end": 5.0}],
    }
    row.update(row_overrides)
    if tail_gap is not None:
        row["tail_gap"] = tail_gap
    calls = []
    monkeypatch.setattr(
        td, "_load_window",
        lambda p, s, d, sr: (calls.append((round(s, 6), round(d, 6))), np.zeros(1, np.float32))[1],
    )  # fmt: skip
    loader = _text_loader(tmp_path, [row], [])
    loader.rng = _PickCut(1)
    loader._sample(loader.get_entry(0))
    return next(d for s, d in calls if s == round(100.0 + 2.1, 6))


def test_a_row_without_tail_gap_is_trimmed_as_before(tmp_path, monkeypatch):
    # The row ends with the last word, so there is nothing past it to keep.
    assert _target_seconds(tmp_path, monkeypatch, None) == pytest.approx(5.0 - 2.1)


@pytest.mark.parametrize(("tail_gap", "tail"), [(0, 0.0), (100, 0.05), (400, 0.2), (3000, 0.2)])
def test_the_tail_read_past_the_row_is_capped_and_at_most_half_the_silence(
    tmp_path, monkeypatch, tail_gap, tail
):
    seconds = _target_seconds(tmp_path, monkeypatch, tail_gap)
    assert seconds == pytest.approx(5.0 + tail - 2.1)


def test_the_tail_extends_past_the_rows_own_duration(tmp_path, monkeypatch):
    seconds = _target_seconds(tmp_path, monkeypatch, 3000)
    assert seconds > 5.0 - 2.1


def test_get_entry_reads_tail_gap(tmp_path):
    row = dict(_fixture_rows()[0], tail_gap=240)
    assert _text_loader(tmp_path, [row], []).get_entry(0).tail_gap == 240
    assert _text_loader(tmp_path, _fixture_rows()[:1], []).get_entry(0).tail_gap is None
