"""
Build parallel CSVs for each translation direction from an ALREADY
DOWNLOADED local copy of FLORES-200.
 
I downloaded flores200_dataset.tar.gz manually via the browser and
extracted it, so this script skips downloading entirely and just reads
straight from my local folder.

Usage (Windows PowerShell example):
    python build_flores_csvs.py "C:\\Users\\sofoh\\Downloads\\flores200_dataset\\flores200_dataset"

Usage (Mac/Linux example):
    python build_flores_csvs.py /path/to/flores200_dataset

The path you pass in should be the folder that DIRECTLY contains
'dev/', 'devtest/', 'metadata_dev.tsv', etc. -- i.e. the inner folder,
since the zip extracted into a folder with the same name nested inside
itself.
"""

import os
import sys
import csv

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FLORES-200 codes for the languages in this study
FLORES_LANGS = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "sw": "swh_Latn",
    "zu": "zul_Latn",
}

# Directions required by the research proposal (§4.8)
DIRECTIONS = [
    ("en", "fr"),
    ("en", "sw"),
    ("en", "zu"),
    ("sw", "en"),
    ("zu", "en"),
]


def load_sentences(root, split, flores_code):
    """
    FLORES-200 layout: {root}/{split}/{flores_code}.{split}
    e.g. flores200_dataset/dev/eng_Latn.dev
    """
    path = os.path.join(root, split, f"{flores_code}.{split}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path} -- double check the root folder you "
            f"passed in is the one that directly contains 'dev/' and "
            f"'devtest/' subfolders."
        )
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_direction_csvs(root):
    print(f"Reading FLORES-200 from: {root}\n")
    sentences = {}
    for short_code, flores_code in FLORES_LANGS.items():
        sentences[short_code] = {
            "dev": load_sentences(root, "dev", flores_code),
            "devtest": load_sentences(root, "devtest", flores_code),
        }
        print(f"  Loaded {flores_code}: "
              f"{len(sentences[short_code]['dev'])} dev, "
              f"{len(sentences[short_code]['devtest'])} devtest sentences")

    print("\nBuilding parallel CSVs per translation direction...")
    for src, tgt in DIRECTIONS:
        for split in ["dev", "devtest"]:
            src_sents = sentences[src][split]
            tgt_sents = sentences[tgt][split]
            assert len(src_sents) == len(tgt_sents), \
                "FLORES-200 splits should be sentence-aligned across languages"

            out_path = os.path.join(OUTPUT_DIR, f"flores200_{src}-{tgt}_{split}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "target"])
                writer.writerows(zip(src_sents, tgt_sents))
            print(f"  Wrote {out_path} ({len(src_sents)} sentence pairs)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_flores_csvs.py <path_to_flores200_dataset_folder>")
        sys.exit(1)

    root = sys.argv[1]
    build_direction_csvs(root)
    print("\nDone. Parallel CSVs are in the 'data/' directory, "
          "one per translation direction and split.")
