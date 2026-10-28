"""Script registry: fonts, corpora and (optional) Unicode transliteration.

Adding a new script only needs a block in config.yaml: one or more fonts and a
corpus. If the corpus is written in another Brahmic script, set
`transliterate: {from: DEVANAGARI, to: <UNICODE BLOCK PREFIX>}` and the text is
mapped by Unicode character *names* (e.g. DEVANAGARI LETTER KA -> SHARADA LETTER KA).
"""
import random
import unicodedata as ud
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont

from .config import resolve


@lru_cache(maxsize=None)
def build_translit_table(src, dst):
    table = {}
    for cp in range(0x0900, 0x11700):
        ch = chr(cp)
        try:
            name = ud.name(ch)
        except ValueError:
            continue
        if not name.startswith(src + " "):
            continue
        try:
            table[ch] = ud.lookup(dst + " " + name[len(src) + 1:])
        except KeyError:
            pass
    return table


def transliterate(text, src, dst):
    table = build_translit_table(src, dst)
    out = []
    for ch in ud.normalize("NFD", text):
        if ch in table:
            out.append(table[ch])
        elif ch in " \t":
            out.append(" ")
        elif ch.isdigit() and ud.name(ch, "").startswith(src + " DIGIT"):
            out.append(table.get(ch, ""))
    return "".join(out)


@lru_cache(maxsize=None)
def font_codepoints(path):
    f = TTFont(path, lazy=True)
    return frozenset(f.getBestCmap().keys())


def convert_digits(s, script_cfg):
    """Latin digits -> digits of the target script (via the script's digit block)."""
    prefix = (script_cfg.get("transliterate") or {}).get("to", script_cfg.get("digit_block", "DEVANAGARI"))
    out = []
    for ch in s:
        if ch.isdigit() and ch in "0123456789":
            name = ud.name(ch).replace("DIGIT", prefix + " DIGIT")
            try:
                out.append(ud.lookup(name))
            except KeyError:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


class ScriptSpec:
    def __init__(self, name, scfg, cfg):
        self.name = name
        self.cfg = scfg
        self.fonts = []
        for fe in scfg["fonts"]:
            fe = {"path": fe} if isinstance(fe, str) else dict(fe)
            fe["path"] = str(resolve(cfg, fe["path"]))
            fe.setdefault("size_scale", 1.0)
            fe.setdefault("weight", 0.0)
            self.fonts.append(fe)
        self.prefix = (scfg.get("transliterate") or {}).get("to", scfg.get("digit_block", "DEVANAGARI"))
        self.danda = self._lookup("DOUBLE DANDA", "|")
        self.danda1 = self._lookup("DANDA", "|")
        self.units = self._load_corpus(cfg, scfg)
        self.headings = self._load_headings(scfg)

    def _lookup(self, suffix, default):
        try:
            return ud.lookup(f"{self.prefix} {suffix}")
        except KeyError:
            return default

    def _load_headings(self, scfg):
        tr = scfg.get("transliterate")
        supported = set.intersection(*[set(font_codepoints(f["path"])) for f in self.fonts])
        out = []
        for h in scfg.get("headings", []):
            h = transliterate(h, tr["from"], tr["to"]) if tr else h
            h = "".join(c for c in h if c == " " or ord(c) in supported)
            if h.strip():
                out.append(h.strip())
        return out

    # ------------------------------------------------------------------
    def _load_corpus(self, cfg, scfg):
        tr = scfg.get("transliterate")
        supported = None
        if self.fonts:
            supported = set.intersection(*[set(font_codepoints(f["path"])) for f in self.fonts])
        sources = []
        for p in scfg["corpus"]:
            path = resolve(cfg, p)
            units = []
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                if tr:
                    line = transliterate(line, tr["from"], tr["to"])
                if supported is not None:
                    line = "".join(c for c in line if c == " " or ord(c) in supported)
                line = " ".join(line.split())
                if len(line) > 6:
                    units.append(line)
            if units:
                sources.append(units)
        if not sources:
            raise ValueError(f"No usable corpus text for script {self.name}")
        return sources  # list of corpora (each a list of verse-lines)

    # ------------------------------------------------------------------
    def token_stream(self, rng, verse_numbers=True):
        """Infinite generator of {'text','role'} tokens (words + verse markers)."""
        corpus = self.units[int(rng.integers(len(self.units)))]
        i = int(rng.integers(len(corpus)))
        num = int(rng.integers(1, 300))
        marks = set(self.danda + self.danda1 + "|")
        while True:
            words = corpus[i % len(corpus)].split()
            if verse_numbers and words:         # ||12|| replaces the closing double danda
                last = words[-1].rstrip(self.danda)
                words = words[:-1] + ([last] if last else [])
            for w in words:
                yield {"text": w, "role": "marker" if set(w) <= marks else "word"}
            if verse_numbers:
                n = convert_digits(str(num), self.cfg)
                yield {"text": f"{self.danda}{n}{self.danda}", "role": "marker"}
                num += 1
            i += 1

    def sample_words(self, rng, n):
        """A few plain words for marginal notes."""
        corpus = self.units[int(rng.integers(len(self.units)))]
        line = corpus[int(rng.integers(len(corpus)))].split()
        words = [w for w in line if len(w) > 2 and not set(w) <= set(self.danda + self.danda1)]
        return words[:n] if words else line[:n]
