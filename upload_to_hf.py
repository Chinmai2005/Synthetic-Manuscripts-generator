#!/usr/bin/env python3
"""Write the dataset card and push the generated dataset to a Hugging Face dataset repo.

    huggingface-cli login            # once
    python upload_to_hf.py --repo-id <username>/synthetic-indic-manuscripts [--private]
"""
import argparse
from pathlib import Path

CARD = """---
license: cc-by-4.0
task_categories:
- image-to-text
language:
- mr
- sa
- hi
tags:
- ocr
- synthetic
- manuscripts
- indic
- devanagari
- modi
- sharada
pretty_name: Synthetic Indic Manuscripts
configs:
- config_name: devanagari
  data_dir: devanagari
- config_name: modi
  data_dir: modi
- config_name: sharada
  data_dir: sharada
---

# {name}

Synthetic historical-manuscript folios (aged handmade paper and palm leaf) with synchronised
ground-truth transcriptions, for training OCR / HTR models on Indic scripts.

| Subset | Script | Train | Validation | Test |
|---|---|---|---|---|
| `devanagari` | Devanagari | 85 | 10 | 5 |
| `modi` | Modi | 85 | 10 | 5 |
| `sharada` | Sharada | 85 | 10 | 5 |

```python
from datasets import load_dataset
ds = load_dataset("{repo}", "modi")        # or "devanagari", "sharada"
print(ds["train"][0]["text"])
```

Each split folder contains `Image_<n>.png`, the matching `Image_<n>.md` transcription and a
`metadata.jsonl` (material, layout, font, line bounding boxes, highlighted spans, seed).
Generated with the open-source pipeline described in the code repository README.
Texts are public-domain Marathi / Hindi / Sanskrit verses (Modi and Sharada are transliterated by Unicode character name).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--dir", default="output")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    d = Path(a.dir)
    (d / "README.md").write_text(CARD.format(name=a.repo_id.split("/")[-1], repo=a.repo_id), encoding="utf-8")
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    for sub in ["devanagari", "modi", "sharada"]:
      print("Uploading", sub, "...")
      api.upload_folder(folder_path=str(d / sub), path_in_repo=sub, repo_id=a.repo_id, repo_type="dataset")
      api.upload_file(path_or_fileobj=str(d / "README.md"), path_in_repo="README.md", repo_id=a.repo_id, repo_type="dataset")
    print(f"Uploaded: https://huggingface.co/datasets/{a.repo_id}")


if __name__ == "__main__":
    main()
