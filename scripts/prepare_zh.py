"""Build the Chinese dataset: CHECKED + thunlp/CED_Dataset → zh_combined.csv.

Both sources are Weibo-based, with fake/real labels.

    * CHECKED                — 344 fake + 1760 real microblogs (COVID-19),
                               labels in CSV.
    * thunlp/CED_Dataset     — 1538 rumors + 1849 non-rumors, JSON per
                               microblog, label inferred from which
                               sub-folder its repost file lives in.

Final balanced merge ≈ 1882 fake + 3609 real ≈ 5491 rows (we keep all real
and all fake — no down-sampling, since both sources are independently
labelled. ``src.data_utils.stratified_split`` will preserve class ratio.)

Usage
-----

    python scripts\\prepare_zh.py

After it finishes the file ``data/raw/zh/zh_combined.csv`` will be ready
and the ZH loader will pick it up automatically.
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
ZH_RAW = REPO_ROOT / "data" / "raw" / "zh"
ZH_RAW.mkdir(parents=True, exist_ok=True)

CHECKED_ZIP = "https://codeload.github.com/cyang03/CHECKED/zip/refs/heads/master"
THUNLP_ZIP = "https://codeload.github.com/thunlp/Chinese_Rumor_Dataset/zip/refs/heads/master"


# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already exists: {dest.name}")
        return
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  wrote {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")


def _strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip HTML tags + URLs + collapse whitespace
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
def load_checked() -> pd.DataFrame:
    """Download CHECKED repo zip → extract dataset/{fake_news,real_news}/*.json
    → assemble into a (text, label) DataFrame.

    We use the JSON files (one per microblog) rather than the CSV summaries
    because the CSV files in CHECKED are tracked with Git LFS and
    raw.githubusercontent.com returns 404 for them.
    """
    print("=== CHECKED ===")
    zip_path = ZH_RAW / "CHECKED_repo.zip"
    _download(CHECKED_ZIP, zip_path)

    print("  reading zip …")
    rows: List[Tuple[str, int]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            # paths look like: CHECKED-master/dataset/fake_news/<hash>.json
            #             or:  CHECKED-master/dataset/real_news/<hash>.json
            if "/dataset/fake_news/" in name:
                lbl = 1
            elif "/dataset/real_news/" in name:
                lbl = 0
            else:
                continue
            try:
                obj = json.loads(zf.read(name).decode("utf-8", errors="ignore"))
            except Exception:
                continue
            text = obj.get("text") or obj.get("content") or ""
            text = _strip_html(text)
            if len(text) < 5:
                continue
            rows.append((text, lbl))
    out = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates(subset=["text"])
    print(f"  CHECKED rows: {len(out)}  | fake={int((out['label']==1).sum())}  real={int((out['label']==0).sum())}")
    return out


# ---------------------------------------------------------------------------
def load_ced_dataset() -> pd.DataFrame:
    print("=== thunlp / CED_Dataset ===")
    cache = ZH_RAW / "thunlp_Chinese_Rumor_Dataset.zip"
    _download(THUNLP_ZIP, cache)

    print("  reading zip …")
    with zipfile.ZipFile(cache) as zf:
        names = zf.namelist()

        def filenames_under(prefix: str) -> set[str]:
            return {Path(n).stem for n in names
                    if n.startswith(prefix) and not n.endswith("/")}

        # The label of an "original-microblog/<id>.json" file is determined by
        # whether the same id appears under rumor-repost/ (label=1) or
        # non-rumor-repost/ (label=0).
        rumor_ids = filenames_under("Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost/")
        non_rumor_ids = filenames_under("Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/")

        # original-microblog can be either folder structure: with each id as
        # its own .json, OR as one json-per-line. Handle both.
        orig_prefix = "Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/"
        orig_files = [n for n in names if n.startswith(orig_prefix) and n.endswith(".json")]
        print(f"  rumor ids: {len(rumor_ids)}  non-rumor ids: {len(non_rumor_ids)}  "
              f"original microblogs: {len(orig_files)}")

        rows: List[Tuple[str, int]] = []
        for n in orig_files:
            stem = Path(n).stem
            if stem in rumor_ids:
                label = 1
            elif stem in non_rumor_ids:
                label = 0
            else:
                continue  # unknown — skip
            try:
                obj = json.loads(zf.read(n).decode("utf-8", errors="ignore"))
            except Exception:
                continue
            text = obj.get("text") or obj.get("content") or ""
            text = _strip_html(text)
            if len(text) < 5:
                continue
            rows.append((text, label))

    out = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates(subset=["text"])
    print(f"  CED rows: {len(out)}  | fake={int((out['label']==1).sum())}  real={int((out['label']==0).sum())}")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    a = load_checked()
    b = load_ced_dataset()

    print()
    print("=== merge ===")
    df = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df[df["text"].str.len() >= 5]
    df["label"] = df["label"].astype(int)

    print(f"  total rows: {len(df)}")
    print(f"  fake = {int((df['label']==1).sum())}  real = {int((df['label']==0).sum())}")

    out = ZH_RAW / "zh_combined.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    print()
    print("Next: confirm `src/data_utils.py:load_weibo_rumor` reads zh_combined.csv,")
    print("then run `python -m src.train --model baseline --lang zh` for a smoke test.")


if __name__ == "__main__":
    main()
