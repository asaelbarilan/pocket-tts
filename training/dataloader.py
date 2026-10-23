"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.

Rows with `words` are cut at a random word boundary and trained on the words after the cut
(`words` IS the text; `transcript` is only used when no cut exists). Two optional row keys vary
that text per sample, see training/text_augment.py: `alt_words` (rewritten words swapped back
for what the corpus wrote: digits, abbreviations, punctuation, ...) and `nikud` (a vocalized form of each word). Rows without
them behave exactly as before and consume no random numbers.
"""

import json
import logging
import queue
import random
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sphn
import torch
from torchcodec.decoders import AudioDecoder

from pocket_tts.data.audio_utils import convert_audio
from training.text_augment import (
    DEFAULT_ALT_WORD_SWAP_PROB,
    DEFAULT_ALT_WORDS_SAMPLE_PROB,
    DEFAULT_NIKUD_WORD_PROB,
    TextAugment,
    derive_seed,
    render_text,
)

logger = logging.getLogger(__name__)

STOP_POLL_SEC = 0.1  # how often a prefetch worker blocked on a full queue checks for a stop


@dataclass
class Entry:
    path: str
    duration: float
    transcript: str
    words: list | None = None  # [{"word", "start", "end"}] from training.scripts.align_data
    start: float = 0.0  # offset of the utterance inside the audio file (long recordings)
    # {"norms": [{"from", "to", "replace", "k"}]}: spans of `words` that can be swapped for the
    # corpus's own spelling (`k` is a one-letter tag the loader ignores)
    alt_words: dict | None = None
    nikud: list[str] | None = None  # one vocalized string per entry of `words`
    # Seed of this sample's alt_words/nikud draws; None when the row has neither key or both
    # features are off, in which case the plain words are used and nothing is drawn.
    text_seed: int | None = None
    # Seed of this sample's cut generator; set only by a `fixed_draws` loader (validation).
    # None: the cut is drawn from the loader's shared `rng`.
    cut_seed: int | None = None


@dataclass
class Batch:
    audio: torch.Tensor  # [B, 1, samples], zero-padded
    num_audio_frames: torch.Tensor  # [B] valid codec frames per sample
    text_tokens: list[torch.Tensor]  # ragged, one [L_b] long tensor per sample
    voice_audio: torch.Tensor  # [B, 1, prompt_samples]
    num_voice_prompt_frames: torch.Tensor  # [B] valid codec frames of each voice prompt


def load_entries(path: str, rank: int, world_size: int) -> list[str]:
    entries = []
    with Path(path).open() as f:
        for idx, line in enumerate(f):
            if idx % world_size != rank:
                continue
            # We parse lazily, 6x less memory
            entries.append(line)
    logger.info(f"loaded {len(entries)} entries from {path} (rank {rank}/{world_size})")
    assert entries, f"no entries for rank {rank} in {path}"
    return entries


def _load_window(path: str, start_sec: float, duration_sec: float, sample_rate: int) -> np.ndarray:
    wav, sr = _read_audio_window(path, start_sec, duration_sec, sample_rate)
    wav = wav.mean(axis=0)  # mono
    if sr != sample_rate:
        resampled = convert_audio(torch.from_numpy(wav)[None], int(sr), int(sample_rate), 1)
        wav = resampled[0].numpy()
    return wav


def _read_audio_window(
    path: str, start_sec: float, duration_sec: float, sample_rate: int
) -> tuple[np.ndarray, int]:
    suffix = Path(path).suffix.lower()
    if suffix in {".wav", ".ogg", ".opus"}:
        return sphn.read(path, start_sec=start_sec, duration_sec=duration_sec)
    # sphn only decodes wav/ogg; m4a (AAC) and mka (Opus-in-Matroska) are
    # decoded on the fly with torchcodec (FFmpeg) so the source audio is
    # used as-is. Dispatch by extension to avoid a guaranteed sphn failure.
    return _torchcodec_read_window(path, start_sec, duration_sec, sample_rate)


def _torchcodec_read_window(
    path: str, start_sec: float, duration_sec: float, sample_rate: int
) -> tuple[np.ndarray, int]:
    decoder = AudioDecoder(path, sample_rate=sample_rate, num_channels=1)
    samples = decoder.get_samples_played_in_range(
        start_seconds=start_sec, stop_seconds=start_sec + duration_sec
    )
    return samples.data.numpy(), sample_rate


class DataLoader:
    def __init__(
        self,
        jsonl: str,
        tokenize: Callable[[str], list[int]],
        batch_size: int,
        sample_rate: int,
        frame_rate: float,
        max_duration_sec: float,
        max_voice_prompt_sec: float,
        rank: int,
        world_size: int,
        seed: int = 0,
        shuffle: bool = True,
        io_workers: int = 16,
        alt_words_sample_prob: float = DEFAULT_ALT_WORDS_SAMPLE_PROB,
        alt_word_swap_prob: float = DEFAULT_ALT_WORD_SWAP_PROB,
        nikud_word_prob: float = DEFAULT_NIKUD_WORD_PROB,
        fixed_draws: bool = False,
    ):
        """The three probabilities are documented in training/text_augment.py and apply only
        to rows carrying `alt_words` / `nikud` (a ValueError outside [0, 1]).

        `fixed_draws` (validation) makes every random choice for a row, the cut (or the
        fallback prompt window) and the alt_words/nikud draws, a pure function of
        (seed, path, start): the same on every pass, in any row order and however the I/O
        threads are scheduled. Off (training), the cut comes from the shared `self.rng` and
        each text draw gets a fresh seed, so a row varies from epoch to epoch.
        """
        self.jsonl = jsonl
        self.entries = load_entries(jsonl, rank, world_size)
        self.tokenize = tokenize
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.max_duration_sec = max_duration_sec
        self.max_voice_prompt_sec = max_voice_prompt_sec
        self.shuffle = shuffle
        self.io_workers = io_workers
        self._failures = 0
        self.rng = random.Random(seed)
        self.seed = seed
        self.text_aug = TextAugment(alt_words_sample_prob, alt_word_swap_prob, nikud_word_prob)
        self.fixed_draws = fixed_draws
        # A stream of its own, drawn from in `get_entry` only (sequential, outside the
        # pool), so the cut/shuffle stream above is never touched by text augmentation.
        self._text_rng = random.Random(f"text:{seed}")

    text_aug = TextAugment()  # class default for loaders built without __init__ (tests)

    MIN_CUT_SEC = 1.0  # keep at least this much audio on both sides of a cut
    TRAIL_SEC = 0.2  # silence kept after the last word, so EOS has a consistent target

    def _cap_prompt(self, prompt: np.ndarray) -> tuple[np.ndarray, int]:
        """Truncate the prompt to the configured cap; collation pads to batch max."""
        if self.max_voice_prompt_sec > 0:
            prompt_samples = int(self.max_voice_prompt_sec * self.sample_rate)
            prompt = prompt[:prompt_samples]
        return prompt, len(prompt)

    @staticmethod
    def _last_word_end(entry: Entry) -> float | None:
        """End of the last aligned word, i.e. where the speech actually stops.

        None when the entry carries no alignment, which is the documented
        manifest format without `words`.
        """
        if not entry.words:
            return None
        ends = [w["end"] for w in entry.words if w.get("end") is not None]
        return max(ends) if ends else None

    def _words_text(self, entry: Entry, first_word: int) -> str:
        """The text of `words[first_word:]`, with this sample's digit/niqqud variants."""
        return render_text(
            [w["word"] for w in entry.words],
            entry.nikud,
            entry.alt_words,
            first_word,
            entry.text_seed,
            self.text_aug,
        )

    def _sample(self, entry: Entry) -> tuple[np.ndarray, torch.Tensor, np.ndarray, int]:
        # Training draws the cut from the shared `self.rng`. That runs on the pool threads in
        # whatever order they are scheduled, which is fine for training but would hand a
        # validation row a different cut from one evaluation to the next. A `fixed_draws`
        # loader gives each row its own generator instead, so the cut depends on the row only.
        rng = self.rng if entry.cut_seed is None else random.Random(f"cut:{entry.cut_seed}")
        if entry.words:
            # Cut the utterance at a random point between two aligned words:
            # audio before the cut = voice conditioning, audio after = target,
            # paired with the remaining words as text (see training/scripts/align_data.py).
            cuts = []
            for i in range(1, len(entry.words)):
                prev, cur = entry.words[i - 1], entry.words[i]
                if prev["end"] is None or cur["start"] is None:
                    continue
                cut = 0.5 * (prev["end"] + cur["start"])
                if cut >= self.MIN_CUT_SEC and entry.duration - cut >= self.MIN_CUT_SEC:
                    cuts.append((cut, i))
            if cuts:
                # The prompt is the (contiguous) start of the utterance.
                # Eligible cuts are word boundaries whose preceding word ends
                # inside the prompt window; the draw is uniform over 1..k
                # eligible words, so prompt length varies and the target
                # keeps most of the utterance.
                window = (
                    self.max_voice_prompt_sec if self.max_voice_prompt_sec > 0 else float("inf")
                )
                eligible = [
                    (c, i)
                    for c, i in cuts
                    if entry.words[i - 1].get("end") is not None
                    and entry.words[i - 1]["end"] < window
                ]
                cuts = eligible or cuts[:1]  # degenerate windows: earliest valid cut
            if cuts:
                cut, i = rng.choice(cuts)
                text = self._words_text(entry, i)
                tokens = torch.tensor(self.tokenize(text), dtype=torch.long)
                # Trim to the last word (plus a short tail) rather than the end
                # of the file: 12% of utterances carry >1s of trailing silence,
                # and training on it teaches the model to emit silence instead of
                # EOS, so generations never terminate. dora does the same thing
                # via align_wav_on_words.
                end = entry.duration
                last = self._last_word_end(entry)
                if last is not None and last > cut:
                    end = min(entry.duration, last + self.TRAIL_SEC)
                wav = _load_window(
                    entry.path,
                    entry.start + cut,
                    min(end - cut, self.max_duration_sec),
                    self.sample_rate,
                )
                # The prompt is everything from the utterance start to the
                # cut (bounded by the window via cut selection).
                prompt, length = self._cap_prompt(
                    _load_window(entry.path, entry.start, cut, self.sample_rate)
                )
                return wav, tokens, prompt, length
        last = self._last_word_end(entry)
        end = min(entry.duration, last + self.TRAIL_SEC) if last is not None else entry.duration
        duration = min(end, self.max_duration_sec)
        wav = _load_window(entry.path, entry.start, duration, self.sample_rate)
        # Rows with digit/niqqud variants take their text from `words` (cut at word 0), the
        # same text a cut sample would get; every other row keeps its transcript as before.
        use_words = entry.words and entry.text_seed is not None
        text = self._words_text(entry, 0) if use_words else entry.transcript
        tokens = torch.tensor(self.tokenize(text), dtype=torch.long)
        # No alignment: voice prompt from a random window of the same file.
        fallback_sec = self.max_voice_prompt_sec if self.max_voice_prompt_sec > 0 else 3.0
        prompt_start = rng.uniform(0, max(0.0, entry.duration - fallback_sec))
        prompt, length = self._cap_prompt(
            _load_window(entry.path, entry.start + prompt_start, fallback_sec, self.sample_rate)
        )
        return wav, tokens, prompt, length

    def get_entry(self, index: int) -> Entry:
        d = json.loads(self.entries[index])
        start = float(d.get("start", 0.0))
        words = d.get("words")
        alt_words, nikud = d.get("alt_words"), d.get("nikud")
        text_seed = cut_seed = None
        if self.fixed_draws:
            cut_seed = derive_seed(self.seed, d["path"], start)
        # Only rows with something to vary (and only while a feature that applies to them is
        # on) get a seed, so every other row draws nothing, bit-identical to before.
        if words and self.text_aug.wants(bool(alt_words), bool(nikud)):
            text_seed = cut_seed if self.fixed_draws else self._text_rng.getrandbits(63)
        return Entry(
            d["path"],
            float(d["duration"]),
            d["transcript"],
            words,
            start,
            alt_words,
            nikud,
            text_seed,
            cut_seed,
        )

    def _sample_or_none(self, entry: Entry) -> tuple | None:
        try:
            return self._sample(entry)
        except Exception as exc:  # noqa: BLE001 — skip unreadable samples, whatever the cause
            self._failures += 1
            if self._failures % 1000 == 1:
                logger.warning(f"skipping unreadable sample ({self._failures} so far): {exc}")
            return None

    def __iter__(self) -> Iterator[Batch]:
        # Batches are produced in a background thread (the loader is IO-bound)
        # so the GPU never waits on network storage.
        return _prefetch(self._batches())

    def _batches(self) -> Iterator[Batch]:
        pool = ThreadPoolExecutor(max_workers=self.io_workers)
        try:
            yield from self._batches_on(pool)
        finally:
            # Also when the consumer stops early (generator closed) or a check below raises.
            pool.shutdown(wait=False, cancel_futures=True)

    def _batches_on(self, pool: ThreadPoolExecutor) -> Iterator[Batch]:
        self._failures = 0
        # Each sample is two small reads from network storage, so the loader is
        # latency-bound rather than CPU-bound: fetching a batch's samples
        # concurrently keeps the GPUs fed. sphn releases the GIL, so threads are
        # enough. Without this a cold, wide corpus starves training (~2x).
        if len(self.entries) < self.batch_size:
            raise ValueError(
                f"{len(self.entries)} usable entries for this rank but batch_size="
                f"{self.batch_size}: a batch can never be filled. Lower batch_size, "
                "use fewer ranks, or point at a larger manifest."
            )
        while True:
            yielded = 0
            order = list(range(len(self.entries)))
            if self.shuffle:
                self.rng.shuffle(order)
            samples = []
            for chunk_start in range(0, len(order), self.batch_size):
                chunk = order[chunk_start : chunk_start + self.batch_size]
                # Parse sequentially, outside the pool: get_entry is GIL-bound
                # (unlike sphn's audio reads), so parsing it on the worker
                # threads just contends with itself instead of overlapping.
                chunk_entries = [self.get_entry(i) for i in chunk]
                got = [s for s in pool.map(self._sample_or_none, chunk_entries) if s is not None]
                samples.extend(got)
                if len(samples) < self.batch_size:
                    continue
                batch, samples = samples[: self.batch_size], samples[self.batch_size :]
                yielded += 1
                wavs, tokens, prompts, prompt_lens = zip(*batch, strict=True)
                max_len = max(len(w) for w in wavs)
                audio = torch.zeros(len(wavs), 1, max_len)
                frames = torch.zeros(len(wavs), dtype=torch.long)
                for b, w in enumerate(wavs):
                    audio[b, 0, : len(w)] = torch.from_numpy(w)
                    frames[b] = max(1, int(len(w) * self.frame_rate / self.sample_rate))
                max_prompt = max(len(p) for p in prompts)
                voice = torch.zeros(len(prompts), 1, max_prompt)
                for b, pr in enumerate(prompts):
                    voice[b, 0, : len(pr)] = torch.from_numpy(pr)
                num_voice_prompt_frames = torch.tensor(
                    [max(1, int(n * self.frame_rate / self.sample_rate)) for n in prompt_lens],
                    dtype=torch.long,
                )
                yield Batch(audio, frames, list(tokens), voice, num_voice_prompt_frames)
            if not yielded:
                raise ValueError(
                    f"no readable samples in {self.jsonl}: every entry failed to load "
                    f"({self._failures} failures). Check the paths in the manifest."
                )


def _prefetch(iterator: Iterator[Batch], depth: int = 4) -> Iterator[Batch]:
    """Run the (synchronous, IO-bound) loader in a background thread.

    Closing the returned generator (`contextlib.closing`, or just dropping the last reference)
    stops the thread: it sets `stop`, the worker notices within `STOP_POLL_SEC` even when
    blocked on a full queue, closes the loader's own generator from its own thread (which
    shuts down the loader's I/O pool) and exits. Without that, a consumer that stops early,
    as validation does after `num_valid_batches`, leaves a thread blocked in `q.put` forever.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item) -> bool:
        """False if the consumer is gone, so the worker should stop instead of waiting."""
        while not stop.is_set():
            try:
                q.put(item, timeout=STOP_POLL_SEC)
            except queue.Full:
                continue
            return True
        return False

    def worker():
        # An exception must reach the consumer: if this thread just died, `q.get()` below
        # would wait forever on an end marker that never comes.
        try:
            for item in iterator:
                if not put(item):
                    return
        except BaseException as exc:  # noqa: BLE001 — re-raised in the consuming thread
            put(_WorkerFailure(exc))
            return
        else:
            put(None)
        finally:
            # A generator can only be closed by the thread running it, which is this one.
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    threading.Thread(target=worker, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, _WorkerFailure):
                raise item.exc
            yield item
    finally:
        stop.set()
        while True:  # free the batches still queued
            try:
                q.get_nowait()
            except queue.Empty:
                break


@dataclass
class _WorkerFailure:
    exc: BaseException


@torch.no_grad()
def encode_batch(mimi, batch, device):
    audio = batch.audio.to(device)
    latents = mimi.encode_to_latent(audio)  # [B, T, C]
    T = latents.shape[1]
    num_audio_frames = batch.num_audio_frames.to(device).clamp(max=T)
    mask = torch.arange(T, device=device)[None, :] < num_audio_frames[:, None]
    voice_prompt_latents = mimi.encode_to_latent(batch.voice_audio.to(device))
    num_voice_prompt_frames = batch.num_voice_prompt_frames.to(device).clamp(
        max=voice_prompt_latents.shape[1]
    )
    return latents.float(), mask, voice_prompt_latents.float(), num_voice_prompt_frames
