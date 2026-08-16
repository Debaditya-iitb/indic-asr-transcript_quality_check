#!/usr/bin/env python3
"""
ctc_forced_alignment_indic_conformer.py
=========================================

Batch CTC forced-alignment quality checker for the AI4Bharat
indic-conformer-600m-multilingual NeMo model.

KEY CHANGES vs previous version
-------------------------------
PRIMARY FIX (this version) - LANGUAGE-AWARE LOGIT SLICING.
The IndicConformer 600M has a MULTI-SOFTMAX architecture:
  * 22 languages × 256 tokens each = 5632, plus 1 shared blank → 5633 outputs.
  * Tokens for language L live in the contiguous block
    [L_index * 256, L_index * 256 + 255], with blank at index 5632.
  * MultilingualTokenizer.text_to_ids(text, lang) returns LOCAL ids
    (0..255 within that language's own SentencePiece model) — NOT
    offsetted global ids.
  * Calling ctc_decoder(encoder_output=encoded) without telling it
    the language returns the unmasked joint output — the argmax then
    picks freely across ALL 22 language blocks (greedy decodes as a
    salad of unrelated languages).
  * Combining the two: forced_align gets target id 27 (Marathi-local
    "जे") and aligns it to global index 27 (Assamese, position 27).
    Every reference token is forced onto a wrong column → huge
    negative forced log-probs, score floor of ~ -14 on clean data.

Fix: discover the language's offset in the joint vocab (12 × 256 = 3072
for Marathi), slice the [T, 5633] log-prob tensor down to that
language's 256 token columns + the blank column, re-log_softmax the
slice so it's a proper distribution. Local target ids (0..255) now
index the sliced tensor correctly. Greedy and forced both operate in
the same 257-dim language-specific space, so the LR score behaves the
way it should (close to 0 on matched data, much more negative on
mismatches).

PREVIOUS-RELEASE FIXES STILL IN PLACE
-------------------------------------
1. LR scoring instead of forced-path-only:
       score = (forced_logprob - greedy_logprob) / L
2. Per-token span scores (mean/min).
3. NeMo window_stride is in seconds, not samples.
4. Blank id from decoder.num_classes_with_blank - 1.
5. Exact log-softmax detection via per-frame logsumexp.
6. BOS/EOS stripping uses tokenizer.bos_id / eos_id.
7. Target-id validation; adjacent-repeat-aware minimum frame check.
8. Greedy decode reuses the alignment's log-probs.

NEW IN THIS RELEASE — WADA-SNR PRE-FILTER
-----------------------------------------
Every utterance now gets a WADA-SNR estimate computed from the
(segment-extracted) waveform BEFORE the CTC alignment runs. The value
is propagated as the `snr` column in the per-row report and is
written out to BOTH a TSV (`quality_report.tsv`, as before) and a
CSV (`quality_report.csv`, new). SNR is reported even when alignment
fails for any other reason, so it can be used as an independent
diagnostic.

Reference: C. Kim & R. M. Stern, "Robust Signal-to-Noise Ratio
Estimation Based on Waveform Amplitude Distribution Analysis,"
Interspeech 2008.

THRESHOLD GUIDANCE
------------------
With the slicing fix, matched utterances should score near 0
(typically -0.05 to -0.5 per token); mismatched utterances drop sharply
(-3 and below). Default --score_threshold -2.0 is a reasonable start;
calibrate on ~50 trusted utts + the same audio with shuffled texts and
pick the threshold separating the two distributions.
Add --min_token_score (e.g. -6.0) to also reject utts with any single
badly-aligned token.
"""

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as taF
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

import os as _os, sys as _sys
_nemo_root = _os.environ.get("NEMO_ROOT")
if _nemo_root and _nemo_root not in _sys.path:
    _sys.path.insert(0, _nemo_root)
    log.info("Prepended NEMO_ROOT=%s to sys.path", _nemo_root)

try:
    import nemo.collections.asr as _nemo_asr
except ImportError as _nemo_err:
    raise SystemExit(
        f"\n[ERROR] NeMo failed to import:\n  {_nemo_err}\n\n"
        "Set NEMO_ROOT (or PYTHONPATH) to your NeMo source root, or do an\n"
        "editable install: pip install -e /path/to/NeMo --no-deps "
        "--no-build-isolation\n"
    ) from _nemo_err


# ============================================================================
#  WADA-SNR
# ============================================================================

# Pre-computed G(alpha=0.4) lookup table from Kim & Stern (2008).
# Monotonically increasing from ~0.4097 (SNR = -20 dB) to ~0.4169 (SNR = 100 dB).
# Exactly 121 entries for db_vals = -20, -19, ..., 100.
_WADA_DB_VALS = np.arange(-20, 101, dtype=np.float64)
_WADA_G_VALS = np.array([
    0.40974774, 0.40986926, 0.40998566, 0.41010534, 0.4102188,
    0.4103374,  0.41045077, 0.4105655,  0.4106772,  0.4107877,
    0.4108982,  0.41100943, 0.41111982, 0.41122933, 0.41133794,
    0.41144565, 0.41155243, 0.41165827, 0.41176314, 0.41186705,
    0.41196995, 0.41207185, 0.41217272, 0.41227254, 0.4123713,
    0.412469,   0.41256564, 0.4126612,  0.41275568, 0.41284908,
    0.41294137, 0.41303257, 0.41312267, 0.41321166, 0.41329953,
    0.4133863,  0.41347195, 0.41355647, 0.41363988, 0.41372216,
    0.41380334, 0.41388337, 0.41396229, 0.41404008, 0.41411675,
    0.4141923,  0.41426672, 0.41434003, 0.41441221, 0.41448327,
    0.41455322, 0.41462205, 0.41468976, 0.41475636, 0.41482186,
    0.41488624, 0.41494953, 0.41501172, 0.41507282, 0.41513283,
    0.41519176, 0.4152496,  0.41530637, 0.41536206, 0.41541668,
    0.41547023, 0.41552272, 0.41557414, 0.41562451, 0.41567382,
    0.41572209, 0.41576931, 0.41581549, 0.41586064, 0.41590475,
    0.41594783, 0.41598989, 0.41603093, 0.41607095, 0.41610996,
    0.41614796, 0.41618496, 0.41622096, 0.41625597, 0.41628999,
    0.41632303, 0.41635508, 0.41638616, 0.41641626, 0.41644540,
    0.41647358, 0.41650080, 0.41652708, 0.41655241, 0.41657681,
    0.41660028, 0.41662283, 0.41664447, 0.41666521, 0.41668506,
    0.41670402, 0.41672211, 0.41673933, 0.41675570, 0.41677122,
    0.41678590, 0.41679975, 0.41681279, 0.41682502, 0.41683645,
    0.41684709, 0.41685695, 0.41686604, 0.41687437, 0.41688195,
    0.41688880, 0.41689491, 0.41690031, 0.41690500, 0.41690899,
    0.41691231,
], dtype=np.float64)
assert _WADA_G_VALS.size == _WADA_DB_VALS.size == 121


def wada_snr(wav):
    """
    Blind WADA-SNR estimate in dB from a 1-D waveform.

    Reference:
      C. Kim and R. M. Stern, "Robust Signal-to-Noise Ratio Estimation
      Based on Waveform Amplitude Distribution Analysis," Interspeech 2008.

    Accepts a numpy array or torch tensor of any amplitude range
    (the algorithm is scale-invariant). Returns float (dB), or NaN
    if the signal is empty / all-zero / not finite.
    """
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float64).ravel()

    if wav.size == 0 or not np.isfinite(wav).any():
        return float("nan")

    eps = 1e-10
    abs_wav = np.abs(wav)
    abs_wav[abs_wav < eps] = eps

    v1 = max(eps, float(abs_wav.mean()))
    v2 = float(np.log(abs_wav).mean())
    v3 = math.log(v1) - v2

    g_vals = _WADA_G_VALS
    db_vals = _WADA_DB_VALS

    # Table interpolation (clipped at the ends)
    if v3 <= g_vals[0]:
        return float(db_vals[0])
    if v3 >= g_vals[-1]:
        return float(db_vals[-1])

    idx = int(np.searchsorted(g_vals, v3) - 1)
    idx = max(0, min(idx, len(g_vals) - 2))
    lo, hi = g_vals[idx], g_vals[idx + 1]
    snr_db = db_vals[idx] + (v3 - lo) / (hi - lo) * (db_vals[idx + 1] - db_vals[idx])
    return float(snr_db)


# ============================================================================
#  Backend
# ============================================================================

class IndicConformerBackend:
    """
    Wraps ai4bharat/indic-conformer-600m-multilingual (NeMo hybrid
    RNNT-CTC) and produces log-probability tensors [1, T, V_eff]
    suitable for torchaudio.functional.forced_align.

    For multilingual multi-softmax models, V_eff = per_lang_vocab + 1
    (256 + 1 for IndicConformer) — the output is sliced to the target
    language's block plus the shared blank, then re-normalised.
    """

    def __init__(self, model_path, device="cpu",
                 language=None, blank_id=None, debug=False):

        ta_ver = torchaudio.__version__
        maj, mnr = (int(x) for x in ta_ver.split(".")[:2])
        if (maj, mnr) < (2, 1):
            raise ImportError(
                f"torchaudio {ta_ver} does not have forced_align. "
                "Upgrade: pip install 'torchaudio>=2.1' --no-deps"
            )

        self.device   = torch.device(device)
        self.language = language
        self.debug    = debug
        self._is_logprob = None

        log.info("Loading IndicConformer from %r …", model_path)
        self.model = self._load_nemo(_nemo_asr, model_path)
        self.model.eval()
        self.model.to(self.device)
        log.info("Model ready on %s.", self.device)

        # Hybrid model: ctc_decoder is the CTC head, .decoder is the
        # RNN-T prediction network — pick ctc_decoder.
        self._ctc_dec = getattr(self.model, "ctc_decoder", None) or \
                        getattr(self.model, "decoder", None)
        if self._ctc_dec is None:
            raise RuntimeError("Model has no ctc_decoder or decoder attribute.")
        log.info("CTC decoder type : %s", type(self._ctc_dec).__name__)

        self.tokenizer = self.model.tokenizer
        self._build_vocab()

        self._probe_outputs()
        self._raw_blank_id = None
        if blank_id is not None:
            self._raw_blank_id = int(blank_id)
            log.info("Raw blank ID manually set to %d", self._raw_blank_id)
        else:
            self._find_raw_blank()

        if not (0 <= self._raw_blank_id < self._raw_num_classes):
            raise RuntimeError(
                f"blank_id={self._raw_blank_id} outside [0, "
                f"{self._raw_num_classes})."
            )

        # Language-aware slicing.
        self._lang_offset     = None
        self._lang_vocab_size = None
        if self.language is not None:
            self._setup_language_slice(self.language)
        else:
            if self._raw_num_classes > 1000:
                log.warning(
                    "Model output width is %d (looks multilingual) but "
                    "--language was not provided. Alignment will use the "
                    "full unmasked output, which is almost certainly "
                    "wrong for IndicConformer.", self._raw_num_classes)

        if self._lang_offset is not None:
            self.num_classes = self._lang_vocab_size + 1
            self.blank_id    = self._lang_vocab_size
        else:
            self.num_classes = self._raw_num_classes
            self.blank_id    = self._raw_blank_id

        self.index_duration = self._get_frame_shift()

        log.info("Raw decoder out : %d  (joint, incl. blank)",
                 self._raw_num_classes)
        log.info("Raw blank id    : %d", self._raw_blank_id)
        if self._lang_offset is not None:
            log.info("Language slice  : [%d : %d]  + blank @ %d",
                     self._lang_offset,
                     self._lang_offset + self._lang_vocab_size,
                     self._raw_blank_id)
        log.info("Effective vocab : %d  (blank @ %d)",
                 self.num_classes, self.blank_id)
        log.info("Frame duration  : %.1f ms  (%.1f fps)",
                 self.index_duration * 1000, 1.0 / self.index_duration)

    @staticmethod
    def _load_nemo(nemo_asr, path):
        is_local = str(path).endswith(".nemo")
        errors   = []
        candidates = []
        if hasattr(nemo_asr.models, "EncDecHybridRNNTCTCBPEModel"):
            candidates.append(nemo_asr.models.EncDecHybridRNNTCTCBPEModel)
        candidates += [nemo_asr.models.EncDecCTCModelBPE,
                       nemo_asr.models.EncDecCTCModel]
        for cls in candidates:
            try:
                if is_local:
                    m = cls.restore_from(str(path), map_location="cpu")
                else:
                    m = cls.from_pretrained(str(path), map_location="cpu")
                log.info("Loaded model as %s.", cls.__name__)
                return m
            except Exception as exc:
                errors.append(f"{cls.__name__}: {exc}")
        raise RuntimeError(
            f"Could not load model from {path!r}.\n" + "\n".join(errors))

    def _build_vocab(self):
        tok = self.tokenizer
        log.info("Tokenizer type : %s", type(tok).__name__)
        vocab_size = getattr(tok, "vocab_size", None)
        if vocab_size is not None:
            log.info("Tokeniser vocab_size = %d", vocab_size)
        self.vocab_size = vocab_size or 0
        self.char_list  = [str(i) for i in range(max(self.vocab_size, 1))]

    def _probe_outputs(self):
        with torch.no_grad():
            out = self._raw_decoder_out(torch.zeros(16000))
        self._raw_num_classes = int(out.shape[-1])
        self._probe_frames    = int(out.shape[1])

    def _find_raw_blank(self):
        """NeMo CTC convention: blank is the LAST raw output index."""
        n_wb = getattr(self._ctc_dec, "num_classes_with_blank", None)
        if n_wb:
            self._raw_blank_id = int(n_wb) - 1
            log.info("Blank ID from decoder.num_classes_with_blank: %d",
                     self._raw_blank_id)
            return
        self._raw_blank_id = self._raw_num_classes - 1
        log.info("Blank ID set to last raw output index: %d",
                 self._raw_blank_id)

    def _setup_language_slice(self, language):
        """
        Discover the language's token-block offset and width in the
        joint decoder output. Sets self._lang_offset and
        self._lang_vocab_size. Leaves them None on failure (caller falls
        back to unsliced behaviour and logs a warning).
        """
        tok = self.tokenizer
        offset = None
        per_lang_size = None

        # Path A: explicit offset attribute on the tokeniser
        for attr in ("token_id_offset", "langs_to_offsets",
                     "language_offsets", "lang_offsets", "offsets"):
            m = getattr(tok, attr, None)
            if isinstance(m, dict) and language in m:
                offset = int(m[language])
                log.info("Language offset from tokenizer.%s[%r] = %d",
                         attr, language, offset)
                break

        # Path B: find lang in tokeniser's language list, multiply by
        # per-language vocab width derived from joint output width.
        if offset is None:
            langs_list = None
            for attr in ("langs", "languages", "lang_order"):
                v = getattr(tok, attr, None)
                if isinstance(v, (list, tuple)) and language in v:
                    langs_list = list(v); break
            if langs_list is None:
                for attr in ("tokenizers_dict", "tokenizers",
                             "_tokenizers", "langs_to_tokenizers",
                             "lang_to_tokenizer"):
                    v = getattr(tok, attr, None)
                    if isinstance(v, dict) and language in v:
                        langs_list = list(v.keys()); break

            if langs_list is not None:
                lang_tokens_total = self._raw_num_classes - 1
                if lang_tokens_total % len(langs_list) != 0:
                    log.warning(
                        "Joint token width %d not divisible by %d langs; "
                        "non-uniform per-language vocab is unsupported.",
                        lang_tokens_total, len(langs_list))
                else:
                    per_lang_size = lang_tokens_total // len(langs_list)
                    lang_idx = langs_list.index(language)
                    offset   = lang_idx * per_lang_size
                    log.info("Language %r is index %d/%d → offset = "
                             "%d × %d = %d  (per-lang vocab = %d)",
                             language, lang_idx, len(langs_list),
                             lang_idx, per_lang_size, offset,
                             per_lang_size)

        # Optional refinement: get per-lang vocab from the actual SP model
        if per_lang_size is None and offset is not None:
            for attr in ("tokenizers_dict", "tokenizers", "_tokenizers",
                         "langs_to_tokenizers", "lang_to_tokenizer"):
                v = getattr(tok, attr, None)
                if isinstance(v, dict) and language in v:
                    sub = v[language]
                    sp  = getattr(sub, "tokenizer", sub)
                    if hasattr(sp, "get_piece_size"):
                        per_lang_size = int(sp.get_piece_size()); break
                    if hasattr(sub, "vocab_size"):
                        per_lang_size = int(sub.vocab_size); break
            if per_lang_size is None:
                per_lang_size = 256
                log.warning("Per-language vocab size unknown; assuming 256.")

        if offset is None or per_lang_size is None:
            log.warning(
                "Could not determine slice for language %r. Falling back "
                "to unsliced full output (results will be unreliable on "
                "a multilingual model).", language)
            return

        if offset + per_lang_size > self._raw_num_classes - 1:
            log.warning(
                "Computed slice [%d : %d] overruns the language region "
                "(blank @ %d). Refusing to slice.",
                offset, offset + per_lang_size, self._raw_blank_id)
            return

        self._lang_offset     = int(offset)
        self._lang_vocab_size = int(per_lang_size)

    def _get_frame_shift(self):
        try:
            preproc_cfg   = self.model.preprocessor._cfg
            window_stride = float(preproc_cfg.get("window_stride", 0.01))
            if window_stride > 1.0:
                window_stride = window_stride / 16000.0
            enc_cfg = getattr(self.model.encoder, "_cfg", {})
            subsampling = int(enc_cfg.get("subsampling_factor", 4)) \
                if hasattr(enc_cfg, "get") else 4
            frame_shift = window_stride * subsampling
            log.info("Frame shift: window_stride=%.4f s × subsampling=%d "
                     "→ %.1f ms", window_stride, subsampling,
                     frame_shift * 1000)
            if self._probe_frames:
                empirical = 1.0 / self._probe_frames
                if abs(empirical - frame_shift) / frame_shift > 0.25:
                    log.warning("Config frame shift %.1f ms disagrees "
                                "with probe %.1f ms; using probe.",
                                frame_shift * 1000, empirical * 1000)
                    return empirical
            return frame_shift
        except Exception as e:
            log.warning("Could not compute frame shift from config (%s); "
                        "using probe.", e)
            return 1.0 / self._probe_frames if self._probe_frames else 0.04

    def _raw_decoder_out(self, wav):
        """Preprocessor → encoder → CTC decoder. Returns [1, T, V_raw]."""
        if wav.dim() > 1:
            wav = wav.squeeze()
        wav_in  = wav.float().to(self.device).unsqueeze(0)
        lengths = torch.tensor([wav_in.shape[1]], dtype=torch.long,
                               device=self.device)
        with torch.no_grad():
            processed, proc_len = self.model.preprocessor(
                input_signal=wav_in, length=lengths)
            encoded, enc_len = self.model.encoder(
                audio_signal=processed, length=proc_len)
            out = self._ctc_dec(encoder_output=encoded)
        return out

    def get_log_probs(self, wav):
        """
        Returns [1, T, V_eff] log-probabilities.
        With a language slice, V_eff = per_lang_vocab + 1.
        """
        raw = self._raw_decoder_out(wav)

        if self._is_logprob is None:
            lse = torch.logsumexp(raw[0, 0].float(), dim=-1).item()
            self._is_logprob = abs(lse) < 1e-2
            log.info("Decoder output %s log-softmaxed "
                     "(frame-0 logsumexp = %.4f).",
                     "is already" if self._is_logprob else "is NOT", lse)

        full = raw if self._is_logprob else torch.log_softmax(raw, dim=-1)

        if self._lang_offset is None:
            return full

        o, w = self._lang_offset, self._lang_vocab_size
        tokens = full[:, :, o : o + w]
        blank  = full[:, :, self._raw_blank_id : self._raw_blank_id + 1]
        sliced = torch.cat([tokens, blank], dim=-1)
        # Re-normalise so the slice is a proper distribution
        return torch.log_softmax(sliced, dim=-1)

    def greedy_from_logprobs(self, log_probs):
        pred_ids = log_probs.argmax(-1)[0].cpu().tolist()
        collapsed, prev = [], None
        for idx in pred_ids:
            if idx != prev and idx != self.blank_id:
                collapsed.append(idx)
            prev = idx
        return self.ids_to_readable(collapsed)

    def ids_to_readable(self, ids):
        tok = self.tokenizer
        if self.language is not None:
            try:
                return tok.ids_to_text(ids, self.language)
            except Exception:
                pass
        try:
            return tok.ids_to_text(ids)
        except Exception:
            pass
        try:
            return " ".join(str(t) for t in tok.ids_to_tokens(ids))
        except Exception:
            pass
        return " ".join(str(i) for i in ids)


# ============================================================================
#  Token Converter
# ============================================================================

class TokenConverter:
    def __init__(self, backend, language=None):
        self.backend  = backend
        self.language = language or backend.language
        tok = backend.tokenizer
        self.bos_id = getattr(tok, "bos_id", None)
        self.eos_id = getattr(tok, "eos_id", None)
        if isinstance(self.bos_id, int) and self.bos_id < 0:
            self.bos_id = None
        if isinstance(self.eos_id, int) and self.eos_id < 0:
            self.eos_id = None

    def _text_to_ids(self, text):
        tok = self.backend.tokenizer
        if not hasattr(tok, "text_to_ids"):
            raise RuntimeError(
                f"Tokeniser {type(tok).__name__} has no text_to_ids.")
        try:
            ids = (tok.text_to_ids(text, self.language)
                   if self.language is not None else tok.text_to_ids(text))
        except TypeError:
            ids = tok.text_to_ids(text)
        ids = list(ids)
        if ids and self.bos_id is not None and ids[0] == self.bos_id:
            ids = ids[1:]
        if ids and self.eos_id is not None and ids[-1] == self.eos_id:
            ids = ids[:-1]
        return ids

    def to_token_ids(self, text):
        text = text.strip()
        if not text: return []
        ids = self._text_to_ids(text)
        if self.backend.debug:
            log.debug("Token IDs for '%s': %s", text[:50], ids[:20])
        return ids

    def to_token_string(self, text):
        text = text.strip()
        if not text: return ""
        return self.backend.ids_to_readable(self._text_to_ids(text))


# ============================================================================
#  Aligner
# ============================================================================

class TorchaudioForcedAligner:
    def __init__(self, backend, converter, score_threshold=-2.0,
                 min_token_threshold=None):
        self.backend             = backend
        self.converter           = converter
        self.score_threshold     = score_threshold
        self.min_token_threshold = min_token_threshold
        self._debug_counter      = 0

    @staticmethod
    def _load_audio(path, start=None, end=None):
        if start is not None or end is not None:
            info      = torchaudio.info(path)
            sr_native = info.sample_rate
            f_offset  = int((start or 0.0) * sr_native)
            n_frames  = (max(int(end * sr_native) - f_offset, 1)
                         if end is not None else -1)
            wav, sr = torchaudio.load(path, frame_offset=f_offset,
                                      num_frames=n_frames)
        else:
            wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0) if wav.dim() > 1 else wav.squeeze(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.abs().max() > 1.0:
            wav = wav / 32768.0
        return wav

    @staticmethod
    def _fail(audio_dur, error, ref_tokens="", pred_tokens="", snr=float("nan")):
        return {
            "pass": False, "score": -99.0, "lr_per_frame": -99.0,
            "forced_logprob": -99.0, "greedy_logprob": -99.0,
            "mean_token_score": -99.0, "min_token_score": -99.0,
            "start_time": 0.0, "end_time": 0.0,
            "audio_duration": audio_dur, "target_length": 0,
            "audio_frames": 0,
            "snr": float(snr),
            "ref_tokens": ref_tokens, "pred_tokens": pred_tokens,
            "token_boundaries": [], "error": error,
        }

    def align(self, audio_path, reference_text, start=None, end=None):
        wav       = self._load_audio(audio_path, start=start, end=end)
        audio_dur = len(wav) / 16000.0

        # --- WADA-SNR: blind SNR estimate computed BEFORE CTC alignment ---
        try:
            snr_db = wada_snr(wav)
        except Exception as _snr_exc:
            log.debug("WADA SNR failed: %s", _snr_exc)
            snr_db = float("nan")
        # -----------------------------------------------------------------

        target_ids = self.converter.to_token_ids(reference_text)
        ref_tokens = self.converter.to_token_string(reference_text)
        L          = len(target_ids)
        blank      = self.backend.blank_id
        V          = self.backend.num_classes

        if L == 0:
            return self._fail(audio_dur, "empty target after tokenisation",
                              ref_tokens, snr=snr_db)

        bad = [i for i in target_ids if i < 0 or i >= V or i == blank]
        if bad:
            return self._fail(
                audio_dur,
                f"target ids out of effective range [0, {V}) or == blank "
                f"{blank}: {bad[:5]}", ref_tokens, snr=snr_db)

        try:
            log_probs = self.backend.get_log_probs(wav)
            T = log_probs.shape[1]
            pred_tokens = self.backend.greedy_from_logprobs(log_probs)

            n_repeats  = sum(1 for a, b in zip(target_ids, target_ids[1:])
                             if a == b)
            min_frames = L + n_repeats
            if T < min_frames:
                return self._fail(
                    audio_dur,
                    f"audio too short ({T} frames < {min_frames} needed "
                    f"for {L} tokens)", ref_tokens, pred_tokens, snr=snr_db)

            targets = torch.tensor([target_ids], dtype=torch.long,
                                   device=log_probs.device)
            alignments, scores = taF.forced_align(
                log_probs, targets, blank=blank)

            sc  = scores[0].detach().float().cpu()
            aln = alignments[0].detach().cpu().tolist()

            forced_lp    = sc.sum().item()
            greedy_lp    = log_probs[0].max(dim=-1).values.sum().item()
            lr_score     = (forced_lp - greedy_lp) / L
            lr_per_frame = (forced_lp - greedy_lp) / T

            d = audio_dur / T
            spans = []
            prev, seg_start = blank, 0
            for t, token in enumerate(aln):
                if token != prev:
                    if prev != blank:
                        spans.append((prev, seg_start, t))
                    seg_start, prev = t, token
            if prev != blank:
                spans.append((prev, seg_start, T))

            if len(spans) != L:
                log.warning("Span count %d != target length %d for %s",
                            len(spans), L, audio_path)

            token_scores = [sc[s:e].mean().item() for _, s, e in spans]
            mean_token_score = float(np.mean(token_scores)) if token_scores else -99.0
            min_token_score  = float(np.min(token_scores))  if token_scores else -99.0
            token_boundaries = [(tid, s * d, e * d) for tid, s, e in spans]

            nb = [t for t, a in enumerate(aln) if a != blank]
            start_t = nb[0] * d        if nb else 0.0
            end_t   = (nb[-1] + 1) * d if nb else audio_dur

            ok = lr_score >= self.score_threshold
            if self.min_token_threshold is not None:
                ok = ok and (min_token_score >= self.min_token_threshold)

            if self.backend.debug and self._debug_counter == 0:
                self._debug_counter += 1
                log.info("===== DEBUG: First utterance alignment =====")
                log.info("Audio duration   : %.3f s (%d frames, %.1f ms/frame)",
                         audio_dur, T, d * 1000)
                log.info("WADA-SNR         : %.2f dB", snr_db)
                log.info("Target tokens    : %s", ref_tokens[:100])
                log.info("Target IDs       : %s", target_ids[:20])
                log.info("Greedy decode    : %s", pred_tokens[:100])
                log.info("Effective V/blank: %d / %d", V, blank)
                if self.backend._lang_offset is not None:
                    log.info("Slice            : raw [%d : %d] + blank @ %d",
                             self.backend._lang_offset,
                             self.backend._lang_offset
                                 + self.backend._lang_vocab_size,
                             self.backend._raw_blank_id)
                log.info("log_probs min/max: %.3f / %.3f",
                         log_probs.min().item(), log_probs.max().item())
                log.info("forced_lp        : %.2f", forced_lp)
                log.info("greedy_lp        : %.2f", greedy_lp)
                log.info("lr_score (/tok)  : %.3f   lr/frame: %.4f",
                         lr_score, lr_per_frame)
                log.info("token score mean/min: %.3f / %.3f",
                         mean_token_score, min_token_score)
                log.info("Token boundaries (first 5): %s",
                         token_boundaries[:5])
                log.info("============================================")

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                log.warning("CUDA OOM — skipping %.1f s utterance.", audio_dur)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._fail(audio_dur, "CUDA OOM", ref_tokens, snr=snr_db)
            raise

        return {
            "pass"            : bool(ok),
            "score"           : float(lr_score),
            "lr_per_frame"    : float(lr_per_frame),
            "forced_logprob"  : float(forced_lp),
            "greedy_logprob"  : float(greedy_lp),
            "mean_token_score": mean_token_score,
            "min_token_score" : min_token_score,
            "start_time"      : float(start_t),
            "end_time"        : float(end_t),
            "audio_duration"  : audio_dur,
            "target_length"   : L,
            "audio_frames"    : T,
            "snr"             : float(snr_db),
            "ref_tokens"      : ref_tokens,
            "pred_tokens"     : pred_tokens,
            "token_boundaries": token_boundaries,
            "error"           : "",
        }


# ============================================================================
#  Kaldi I/O
# ============================================================================

def load_kaldi_text(p):
    d = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if parts:
                d[parts[0]] = parts[1] if len(parts) > 1 else ""
    return d

def load_kaldi_wav_scp(p):
    d = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d

def load_kaldi_segments(p):
    d = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                uid, rec, s, e = parts
                d[uid] = (rec, float(s), float(e))
            elif parts:
                log.warning("Ignoring malformed segments line: %r",
                            line.rstrip())
    return d


# ============================================================================
#  Batch loop
# ============================================================================

def run_batch(wav_scp, text, aligner, out_dir, segments=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    utt2text = load_kaldi_text(text)
    utt2wav  = load_kaldi_wav_scp(wav_scp)

    seg_path = segments
    if seg_path is None:
        auto = Path(wav_scp).parent / "segments"
        if auto.exists():
            seg_path = str(auto)
            log.info("Auto-detected segments file: %s", seg_path)

    if seg_path and Path(seg_path).exists():
        utt2seg = load_kaldi_segments(seg_path)
        utt_ids = sorted(set(utt2text) & set(utt2seg))
        log.info("Layout          : segmented corpus")
        log.info("Segments file   : %s", seg_path)
        log.info("Recordings      : %d  (in wav.scp)", len(utt2wav))
        log.info("Utterances      : %d  (text ∩ segments)", len(utt_ids))
    else:
        utt2seg = None
        utt_ids = sorted(set(utt2text) & set(utt2wav))
        log.info("Layout          : one file per utterance")
        log.info("Processing %d utterances.", len(utt_ids))

    if not utt_ids:
        log.error("No utterances found.")
        return

    accepted, rejected, report = [], [], []

    for uid in tqdm(utt_ids, desc="Quality check"):
        try:
            if utt2seg is not None:
                rec, t0, t1 = utt2seg[uid]
                if rec not in utt2wav:
                    r = TorchaudioForcedAligner._fail(
                        t1 - t0, f"recording {rec!r} not in wav.scp")
                else:
                    r = aligner.align(utt2wav[rec], utt2text[uid],
                                      start=t0, end=t1)
                audio_path_col = (f"{utt2wav.get(rec, rec)}"
                                  f"  [{t0:.3f}-{t1:.3f}s]")
            else:
                r = aligner.align(utt2wav[uid], utt2text[uid])
                audio_path_col = utt2wav[uid]
        except Exception as exc:
            log.warning("[%s] unexpected error: %s", uid, exc)
            r = TorchaudioForcedAligner._fail(0.0, str(exc))
            audio_path_col = ""

        row = {"utt_id": uid, "audio_path": audio_path_col,
               "text": utt2text[uid]}
        row.update(r)
        report.append(row)
        (accepted if r["pass"] else rejected).append(uid)

    (out_dir / "accepted_utts.txt").write_text(
        "\n".join(accepted) + "\n", encoding="utf-8")
    (out_dir / "rejected_utts.txt").write_text(
        "\n".join(rejected) + "\n", encoding="utf-8")

    fields = [
        "utt_id", "pass", "score", "lr_per_frame",
        "mean_token_score", "min_token_score",
        "forced_logprob", "greedy_logprob",
        "snr",
        "start_time", "end_time", "audio_duration",
        "target_length", "audio_frames",
        "ref_tokens", "pred_tokens", "error", "text", "audio_path",
    ]

    # TSV (kept for backward compatibility)
    with open(out_dir / "quality_report.tsv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader(); w.writerows(report)

    # CSV (final output with SNR column)
    with open(out_dir / "quality_report.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=",",
                           extrasaction="ignore",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader(); w.writerows(report)

    total  = len(utt_ids)
    n_pass = len(accepted)
    scores = [r["score"] for r in report if r.get("score", -99) > -90]
    snrs   = [r["snr"]   for r in report
              if isinstance(r.get("snr"), float) and np.isfinite(r["snr"])]

    log.info("=" * 52)
    log.info("QUALITY CHECK SUMMARY  (score = LR per token, <= 0)")
    log.info("=" * 52)
    log.info("Total      : %d", total)
    log.info("Accepted   : %d  (%.1f %%)", n_pass, 100 * n_pass / total)
    log.info("Rejected   : %d  (%.1f %%)", total - n_pass,
             100 * (total - n_pass) / total)
    if scores:
        log.info("Mean score : %.3f", np.mean(scores))
        log.info("Median     : %.3f", np.median(scores))
        log.info("P10 / P90  : %.3f / %.3f",
                 np.percentile(scores, 10), np.percentile(scores, 90))
    if snrs:
        log.info("WADA-SNR (dB) — mean: %.2f  median: %.2f  "
                 "P10/P90: %.2f / %.2f",
                 float(np.mean(snrs)), float(np.median(snrs)),
                 float(np.percentile(snrs, 10)),
                 float(np.percentile(snrs, 90)))
    log.info("Report TSV : %s", out_dir / "quality_report.tsv")
    log.info("Report CSV : %s", out_dir / "quality_report.csv")
    log.info("=" * 52)


# ============================================================================
#  CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="CTC forced-alignment QC — IndicConformer 600M")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--segments", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--language", default=None,
                    help="Language code (e.g. 'mr', 'hi'). REQUIRED for "
                         "the multilingual IndicConformer 600M.")
    ap.add_argument("--score_threshold", type=float, default=-2.0)
    ap.add_argument("--min_token_score", type=float, default=None,
                    help="Optional: also reject if any token's mean frame "
                         "log-prob is below this (e.g. -6.0).")
    ap.add_argument("--blank_id", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    backend = IndicConformerBackend(
        args.model, device=args.device, language=args.language,
        blank_id=args.blank_id, debug=args.debug)
    converter = TokenConverter(backend, language=args.language)
    aligner = TorchaudioForcedAligner(
        backend, converter,
        score_threshold=args.score_threshold,
        min_token_threshold=args.min_token_score)
    run_batch(args.wav_scp, args.text, aligner, args.out_dir,
              segments=args.segments)


if __name__ == "__main__":
    main()