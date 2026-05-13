"""One-shot data preparation script.

Reads downloaded archives + datasets and lays them out the way `src/data_utils.py`
expects. Run this **once** after you've downloaded:

    * Russian Troll Tweets (9 IRAhandle_tweets_*.csv files) into one folder
    * Ukrainian Kaggle dataset (data_set_4.csv + news_data.csv) into one folder

It will:

    1. Concatenate IRA tweets, keep only Russian-language rows, save as
       ``data/raw/ru/russian_troll_tweets.csv``.
    2. Combine the two Ukrainian CSVs, normalize columns, save as
       ``data/raw/uk/ufnd.csv``.
    3. Optionally pull a sample of Lenta.ru articles via HuggingFace and
       save as ``data/raw/ru/lenta.csv`` (needs ``datasets`` installed).
    4. Optionally pre-download CHECKED (Chinese) into ``data/raw/zh/``.

Usage:

    cd D:\\news\\ml
    python scripts\\prepare_data.py

You can override defaults with CLI flags — see ``--help``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
def prepare_ru_troll(src_dir: Path, out_path: Path) -> int:
    """Concat IRAhandle_tweets_*.csv, filter language='Russian'."""
    files = sorted(src_dir.glob("IRAhandle_tweets_*.csv"))
    if not files:
        print(f"[RU] no IRAhandle_tweets_*.csv in {src_dir} — skipping")
        return 0

    print(f"[RU] reading {len(files)} files from {src_dir} …")
    chunks = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "language" in df.columns:
            df = df[df["language"].astype(str).str.lower() == "russian"]
        chunks.append(df)
        print(f"     {f.name}: {len(df):>6} russian rows")

    out = pd.concat(chunks, ignore_index=True)
    print(f"[RU] total russian rows: {len(out)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[RU] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return len(out)


def prepare_uk(src_dir: Path, out_path: Path) -> int:
    """Combine the two Ukrainian CSVs into a unified text+label table.

    Label convention in the source files:
        True  → real news (label = 0)
        False → fake news (label = 1)
    """
    files = []
    for name in ("news_data.csv", "data_set_4.csv"):
        p = src_dir / name
        if p.exists():
            files.append(p)
        else:
            print(f"[UK] file not found: {p} — skipping it")

    if not files:
        print(f"[UK] nothing to combine in {src_dir}")
        return 0

    chunks = []
    for f in files:
        df = pd.read_csv(f)
        # Normalize column names — lower-case
        df.columns = [c.strip() for c in df.columns]
        text_col = "Text" if "Text" in df.columns else ("text" if "text" in df.columns else None)
        label_col = "Label" if "Label" in df.columns else ("label" if "label" in df.columns else None)
        if text_col is None or label_col is None:
            print(f"[UK] {f.name}: missing Text/Label, skipping")
            continue
        sub = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        sub = sub.dropna()
        chunks.append(sub)
        print(f"     {f.name}: {len(sub)} rows")

    if not chunks:
        return 0

    out = pd.concat(chunks, ignore_index=True)

    # True → 0 (real), False → 1 (fake)
    label_map = {"True": 0, "False": 1, True: 0, False: 1, "true": 0, "false": 1}
    out["label"] = out["label"].map(label_map).astype("Int64")
    before = len(out)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    if len(out) < before:
        print(f"[UK] dropped {before - len(out)} rows with unrecognized labels")

    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[UK] total: {len(out)} rows | "
          f"real={(out['label']==0).sum()}  fake={(out['label']==1).sum()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[UK] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return len(out)


def prepare_ru_lenta(out_path: Path, n_rows: int = 30_000) -> int:
    """Pull a Lenta.ru sample from HuggingFace as the 'real' side for RU."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[RU/lenta] `datasets` not installed; run `pip install datasets`. Skipping.")
        return 0

    print(f"[RU/lenta] pulling first {n_rows} rows from IlyaGusev/lenta …")
    try:
        ds = load_dataset("IlyaGusev/lenta", split=f"train[:{n_rows}]")
    except Exception as exc:
        print(f"[RU/lenta] failed: {exc}")
        print("           you can also download yutkin's Kaggle Lenta corpus and rename to lenta.csv")
        return 0
    df = ds.to_pandas()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[RU/lenta] wrote {out_path} ({len(df)} rows)")
    return len(df)


def prepare_zh_checked(out_dir: Path) -> int:
    """Download CHECKED fake_news.csv + real_news.csv from raw GitHub."""
    import urllib.request

    base = "https://raw.githubusercontent.com/cyang03/CHECKED/master/dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in ("fake_news.csv", "real_news.csv"):
        target = out_dir / name
        if target.exists():
            print(f"[ZH] {name} already exists, skipping")
            continue
        try:
            print(f"[ZH] downloading {name} …")
            urllib.request.urlretrieve(f"{base}/{name}", target)
            n += 1
        except Exception as exc:
            print(f"[ZH] {name} failed: {exc}")
    return n


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ru-troll-dir",
        default=str(Path.home() / "Downloads" / "russian"),
        help="Folder containing IRAhandle_tweets_*.csv (default: ~/Downloads/russian)",
    )
    ap.add_argument(
        "--uk-dir",
        default=str(Path.home() / "Downloads" / "archive (1)"),
        help="Folder containing the Ukrainian CSVs (default: ~/Downloads/archive (1))",
    )
    ap.add_argument("--lenta-rows", type=int, default=30_000,
                    help="How many Lenta.ru rows to pull (default: 30000)")
    ap.add_argument("--skip-lenta", action="store_true",
                    help="Don't try to download Lenta.ru via HuggingFace")
    ap.add_argument("--skip-zh", action="store_true",
                    help="Don't pre-download CHECKED")
    args = ap.parse_args()

    print(f"Repo root: {REPO_ROOT}")
    print(f"Data dir : {DATA_RAW}")
    print()

    prepare_ru_troll(Path(args.ru_troll_dir), DATA_RAW / "ru" / "russian_troll_tweets.csv")
    print()
    prepare_uk(Path(args.uk_dir), DATA_RAW / "uk" / "ufnd.csv")
    print()
    if not args.skip_lenta:
        prepare_ru_lenta(DATA_RAW / "ru" / "lenta.csv", n_rows=args.lenta_rows)
        print()
    if not args.skip_zh:
        prepare_zh_checked(DATA_RAW / "zh")
        print()

    print("=== done ===")
    print("Next: run `python -m src.train --model baseline --lang en` for a smoke test.")


if __name__ == "__main__":
    main()
