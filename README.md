# Synthetic Indic Manuscript Generator

An end-to-end, fully procedural Python pipeline that turns raw text into realistic **historical manuscript folios**
(aged handmade paper and palm leaf) together with **synchronised Markdown ground truth** for OCR / HTR training.
Supports **Devanagari, Modi and Sharada** out of the box and any other script via `config.yaml`.

- **Hugging Face dataset:** `https://huggingface.co/datasets/Chinmai2005/synthetic-indic-manuscripts`
  (subsets: `devanagari`, `modi`, `sharada`; splits: train 85 / validation 10 / test 5)
- **Code:** this repository

## Pipeline overview

```
raw text (data/corpus/*.txt)
   |  scripts.py   transliteration (Devanagari -> Modi / Sharada by Unicode names), font-coverage filtering
   v
background  ---- backgrounds.py   palm leaf (anisotropic fibres, veins, string holes, splits) / handmade paper
   |                              (fibres, laid lines, foxing, water stains, worm holes, deckled edges, wear)
   v
layout      ---- compose.py       single block, commentary, two-column, marginal-notes, palm-leaf with hole-aware lines;
   |                              invocation heading, ruled frames, guide lines, ornaments, verse markers ||12||,
   |                              red rubrication, highlighted (turmeric/ochre wash) spans, vertical side notes, folio numbers
   v
handwriting ---- handwriting.py   HarfBuzz-shaped words -> per-word pen-pressure thickness, rough edges, slant,
   |                              rotation, elastic wobble; wavy sloping baselines; irregular justification
   v
artefacts   ---- artifacts.py     ink bleeding, faded strokes, dry-pen skips, smudges, blots, show-through
   v
physics     ---- deform.py        surface warp, folds, edge/corner curl, skew + perspective, shading, shadow,
   |                              backdrop, scanner noise / blur / JPEG artefacts
   v
dataset     ---- dataset.py       QA (text must stay inside the folio, no border contact; auto-resample), splits,
                                  Image_<n>.png + Image_<n>.md + metadata.jsonl
```

Text is shaped per word (Raqm/HarfBuzz) so conjuncts and matras are always correct, then every word is distorted
individually, so output looks hand-written rather than typeset while the ground truth stays exact.

## Installation

```bash
git clone <this-repo> && cd manuscript-synth
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```
Pillow wheels ship with libraqm. If you see a warning about Raqm, install `libraqm0` (Debian/Ubuntu: `apt install libraqm0`).
Fonts (SIL OFL) are bundled in `assets/fonts/`.

## Usage

```bash
python generate.py                                   # full dataset: 3 scripts x 100 images -> ./output
python generate.py --workers 4                       # parallel
python generate.py --scripts modi --num 10 --out preview --set image.scale=0.6   # quick preview
python generate.py --seed 7 --set output.markdown_style=structured
python validate.py output                            # counts, pairs, QA flags
python upload_to_hf.py --repo-id <user>/synthetic-indic-manuscripts               # push to Hugging Face
```
Everything is reproducible: each image is seeded from `(seed, script, index, attempt)`.
Generating 300 images takes ~13 minutes on one CPU core (~2.6 s per image).

### Configurable parameters (`config.yaml`)
Materials and palettes, folio sizes, layout weights, handwriting style ranges (slant, waviness, pen pressure, word gaps),
ink and rubrication colours, artefact strengths, deformation (warp, folds, curl, skew), scanner noise, split counts,
image format (`png|jpg|webp`) and annotation style. Any value can be overridden with `--set key.path=value`.

### Adding a new script
Add a block under `scripts:` in `config.yaml` with one or more `fonts`, a `corpus` (text files, one verse/line per row) and,
if the corpus is in another Brahmic script, `transliterate: {from: DEVANAGARI, to: <UNICODE BLOCK NAME>}`
(e.g. `GRANTHA`, `BENGALI`, `GUJARATI`). Characters missing from the font are dropped automatically.
To use your own texts, drop `.txt` files into `data/corpus/` and list them under `corpus:`.

## Repository structure

```
generate.py            entry point (python generate.py)
validate.py            dataset checks
upload_to_hf.py        dataset card + Hugging Face upload
config.yaml            all parameters
requirements.txt
assets/fonts/          OFL fonts (Devanagari calligraphic fonts, Noto Sans Modi, Noto Sans Sharada)
data/corpus/           raw text (Marathi, Hindi/Awadhi, Sanskrit shlokas, Bhagavad Gita)
manuscriptgen/         backgrounds, handwriting, compose, artifacts, deform, scripts, dataset, config, noise
```

## Dataset organisation

```
output/
  devanagari/ | modi/ | sharada/
    train/        Image_1.png  Image_1.md  ...  metadata.jsonl   (85)
    validation/   ...                                            (10)
    test/         ...                                            (5)
```
`Image_<n>.md` contains the transcription line by line in reading order (heading, main block(s), commentary,
marginal notes, folio number), as in the assignment sample. `output.markdown_style: structured` adds section headings and
`==highlighted==` markers. `metadata.jsonl` holds material, layout, font, ink, per-line bounding boxes and highlighted spans.
Hugging Face loading: `load_dataset("<user>/synthetic-indic-manuscripts", "modi")`.

## Notes and limitations
- Modi and Sharada have a single Unicode font each (Noto), so their glyph shapes are print-style and made
  hand-written by distortion; supplying historical-style fonts in `config.yaml` will improve fidelity.
- Modi/Sharada text is machine-transliterated from Marathi/Sanskrit/Hindi sources; use genuine corpora where available.
- "Visual parity" with real folios should be checked against the company's reference scans
  (e.g. FID or an OCR model trained on synthetic and tested on real pages).
