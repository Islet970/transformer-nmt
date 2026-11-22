#!/usr/bin/env python
"""Download IWSLT 2017 zh-en (TED Talks) and export the 6 text files used by this project.

Source
------
HuggingFace dataset ``IWSLT/iwslt2017``, config ``iwslt2017-zh-en``
(https://huggingface.co/datasets/IWSLT/iwslt2017), file::

    data/2017-01-trnted/texts/zh/en/zh-en.zip

The archive is the original WIT3 distribution.  It contains the line-aligned
train files (``train.tags.zh-en.zh/.en``), the dev2010 XML files (validation)
and the tst2010..tst2015 XML files (test).  We flatten them into the layout the
rest of the repo expects::

    <out-dir>/
    ├── train.zh  train.en     # 231,164 pairs
    ├── valid.zh  valid.en     #     879 pairs
    └── test.zh   test.en      #   8,549 pairs

Counts are checked against the numbers documented in ``plan.md``.  Note that the
official tally is 231,266 lines; we drop one TED talk (102 pairs) whose zh/en
line counts disagree, which is what makes the clean count 231,164.

Usage
-----
    python scripts/download_iwslt17.py
    python scripts/download_iwslt17.py --out-dir data/iwslt17_zh_en
    python scripts/download_iwslt17.py --mirror https://hf-mirror.com   # China mirror
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import zipfile

HF_URL = "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/zh/en/zh-en.zip"
MIRROR_URL = "https://hf-mirror.com/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/zh/en/zh-en.zip"

PAIR = "zh-en"
YEARS = [2010, 2011, 2012, 2013, 2014, 2015]
EXPECTED = {"train": 231164, "valid": 879, "test": 8549}


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    print(f"downloading {url}")
    start = time.time()
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 / total
                print(f"\r  {done/1e6:6.1f} / {total/1e6:.1f} MB ({pct:5.1f}%)", end="", flush=True)
    print(f"\r  {done/1e6:6.1f} MB in {time.time()-start:.1f}s" + " " * 20)
    os.replace(tmp, dest)


def extract_text(line: str) -> str:
    """Pull the sentence out of '<seg id="1">text</seg>'; ignore non-seg tags."""
    line = line.strip()
    if not line:
        return ""
    if line.startswith("<"):
        if not line.startswith("<seg"):
            return ""
        line = line.split(">", 1)[1].split("<", 1)[0]
    return line.strip()


def read_xml_lines(zf: zipfile.ZipFile, member: str) -> list[str]:
    with zf.open(member) as fh:
        lines = fh.read().decode("utf-8").splitlines()
    return [text for text in (extract_text(x) for x in lines) if text]


def parse_talks(zf: zipfile.ZipFile, member: str) -> list[dict]:
    """Split train.tags.* into per-talk blocks and keep only the sentence lines.

    The archive interleaves talk metadata (<url>, <title>, <reviewer>, ...) with
    the aligned sentence lines.  Metadata does not always appear in the same
    place in the .zh and .en files, so the files must be paired per talk instead
    of line by line (see main()).
    """
    with zf.open(member) as fh:
        lines = fh.read().decode("utf-8").splitlines()
    talks: list[dict] = []
    cur = None
    for raw in lines:
        text = raw.strip()
        if text.startswith("<url>"):
            if cur is not None:
                talks.append(cur)
            cur = {"url": text, "text": []}
        elif cur is None or not text or text.startswith("<"):
            continue
        else:
            cur["text"].append(text)
    if cur is not None:
        talks.append(cur)
    return talks


def find_member(zf: zipfile.ZipFile, basename: str) -> str:
    for name in zf.namelist():
        if name.replace("\\", "/").endswith(basename):
            return name
    raise KeyError(f"{basename} not found in archive; contents: {zf.namelist()[:20]}")


def write_pairs(path_zh: str, path_en: str, zh: list[str], en: list[str]) -> None:
    assert len(zh) == len(en), f"length mismatch: {len(zh)} vs {len(en)}"
    for path, lines in ((path_zh, zh), (path_en, en)):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join("data", "iwslt17_zh_en"))
    parser.add_argument("--cache-dir", default=os.path.join("data", "_raw"))
    parser.add_argument("--mirror", default="", help="e.g. https://hf-mirror.com (China)")
    parser.add_argument("--force", action="store_true", help="re-download / re-export")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    zip_path = os.path.join(cache_dir, "zh-en.zip")
    marker = os.path.join(out_dir, ".complete")

    if os.path.exists(marker) and not args.force:
        print(f"already exported: {out_dir} (use --force to redo)")
        return 0

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 20_000_000 or args.force:
        urls = [args.mirror.rstrip("/") + "/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/zh/en/zh-en.zip"] if args.mirror else []
        urls.append(HF_URL)
        last_err = None
        for url in urls:
            try:
                download(url, zip_path)
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last_err = exc
                print(f"  failed: {exc}")
        else:
            print(f"ERROR: could not download the archive: {last_err}")
            return 1
    else:
        print(f"using cached archive {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        print(f"archive members: {len(zf.namelist())}")
        splits: dict[str, tuple[list[str], list[str]]] = {}

        zh_talks = parse_talks(zf, find_member(zf, f"train.tags.{PAIR}.zh"))
        en_talks = parse_talks(zf, find_member(zf, f"train.tags.{PAIR}.en"))
        if len(zh_talks) != len(en_talks):
            raise RuntimeError(f"talk count mismatch: {len(zh_talks)} vs {len(en_talks)}")
        train_zh: list[str] = []
        train_en: list[str] = []
        skipped: list[str] = []
        for zh_talk, en_talk in zip(zh_talks, en_talks):
            if zh_talk["url"] != en_talk["url"]:
                raise RuntimeError(f"talk order mismatch: {zh_talk['url']} vs {en_talk['url']}")
            if len(zh_talk["text"]) != len(en_talk["text"]):
                skipped.append(
                    f"{zh_talk['url']} (zh={len(zh_talk['text'])}, en={len(en_talk['text'])})"
                )
                continue
            train_zh += zh_talk["text"]
            train_en += en_talk["text"]
        print(f"talks: {len(zh_talks)}, skipped for misalignment: {len(skipped)}")
        for item in skipped:
            print(f"  skipped {item}")
        splits["train"] = (train_zh, train_en)
        splits["valid"] = (
            read_xml_lines(zf, find_member(zf, f"IWSLT17.TED.dev2010.{PAIR}.zh.xml")),
            read_xml_lines(zf, find_member(zf, f"IWSLT17.TED.dev2010.{PAIR}.en.xml")),
        )
        test_zh: list[str] = []
        test_en: list[str] = []
        for year in YEARS:
            test_zh += read_xml_lines(zf, find_member(zf, f"IWSLT17.TED.tst{year}.{PAIR}.zh.xml"))
            test_en += read_xml_lines(zf, find_member(zf, f"IWSLT17.TED.tst{year}.{PAIR}.en.xml"))
        splits["test"] = (test_zh, test_en)

    ok = True
    for split, (zh, en) in splits.items():
        write_pairs(os.path.join(out_dir, f"{split}.zh"), os.path.join(out_dir, f"{split}.en"), zh, en)
        n = len(zh)
        expected = EXPECTED[split]
        flag = "OK" if n == expected else f"WARNING expected {expected:,}"
        if n != expected:
            ok = False
        print(f"{split:5s}: {n:>7,} pairs  {flag}")
        print(f"       example zh: {zh[0][:50]}")
        print(f"       example en: {en[0][:50]}")

    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("ok\n")
    print(f"\nexported to {out_dir}")
    if not ok:
        print("counts differ from the expected 231,164 / 879 / 8,549.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
