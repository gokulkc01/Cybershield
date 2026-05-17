"""Download and build a MAWI extended_v1 dataset.

This pipeline downloads one or more MAWI samplepoint-F daily traces,
converts each pcap into Zeek logs, and then reuses the extended dataset
builder to produce a 45-feature NPZ artifact.

The script is intentionally conservative: it only depends on the standard
library plus the existing CyberShield build pipeline, so it can run in the
project venv without adding new packages.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

from src.pipelines.build_extended_dataset import build_extended_dataset


MAWI_SAMPLEPOINT_ROOT = "http://mawi.wide.ad.jp/mawi/samplepoint-F"


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _extract_pcap_gz(gz_path: Path) -> Path:
    pcap_path = gz_path.with_suffix("")
    if pcap_path.exists() and pcap_path.stat().st_size > 0:
        return pcap_path

    with gzip.open(gz_path, "rb") as source, pcap_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return pcap_path


def _resolve_day_page(year: str, day_id: str) -> str:
    return f"{MAWI_SAMPLEPOINT_ROOT}/{year}/{day_id}1400.html"


def _resolve_download_url(year: str, day_id: str) -> tuple[str, str]:
    page_url = _resolve_day_page(year, day_id)
    html = _fetch_text(page_url)
    match = re.search(r'href="([^"]+\.pcap\.gz)"', html, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"No pcap.gz link found on MAWI page: {page_url}")
    download_url = match.group(1)
    if download_url.startswith("/"):
        download_url = f"http://mawi.nezu.wide.ad.jp{download_url}"
    return page_url, download_url


def _run_zeek(pcap_path: Path, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    zeek_cmd = shutil.which("zeek")
    if zeek_cmd is None:
        raise RuntimeError(
            "Zeek was not found on PATH. Install Zeek or provide an already extracted conn.log."
        )

    subprocess.run(
        [zeek_cmd, "-Cr", str(pcap_path), "-l", str(log_dir)],
        check=True,
    )

    conn_candidates = list(log_dir.rglob("conn.log"))
    if not conn_candidates:
        raise FileNotFoundError(f"Zeek completed but no conn.log was produced under {log_dir}")
    return conn_candidates[0]


def _prepare_one_day(
    *,
    year: str,
    day_id: str,
    raw_dir: Path,
    work_dir: Path,
    out_root: Path,
) -> dict:
    page_url, download_url = _resolve_download_url(year, day_id)
    print(f"[DOWNLOAD] {page_url}")
    gz_path = raw_dir / f"{day_id}.pcap.gz"
    _download_file(download_url, gz_path)

    print(f"[EXTRACT] {gz_path.name}")
    pcap_path = _extract_pcap_gz(gz_path)

    print(f"[ZEEK] {day_id}")
    log_dir = work_dir / day_id
    conn_log = _run_zeek(pcap_path, log_dir)

    output_dir = out_root / day_id
    manifest = build_extended_dataset(
        conn_log_path=str(conn_log),
        out_dir=str(output_dir),
        source_dataset="mawi",
        label=0,
        family="",
        capture_id=day_id,
        min_flows=1,
        inactivity_timeout=300.0,
    )
    manifest["download_url"] = download_url
    manifest["page_url"] = page_url
    return manifest


def build_mawi_dataset(
    *,
    years: Iterable[str],
    days: Iterable[str],
    raw_dir: str,
    work_dir: str,
    out_dir: str,
) -> list[dict]:
    raw_root = Path(raw_dir)
    work_root = Path(work_dir)
    out_root = Path(out_dir)

    results: list[dict] = []
    for year in years:
        for day in days:
            results.append(
                _prepare_one_day(
                    year=year,
                    day_id=day,
                    raw_dir=raw_root / year,
                    work_dir=work_root / year,
                    out_root=out_root / year,
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build MAWI extended_v1 datasets")
    parser.add_argument(
        "--years",
        default="2025",
        help="Comma-separated MAWI years to process (default: 2025)",
    )
    parser.add_argument(
        "--days",
        default="20250101",
        help="Comma-separated MAWI day ids in YYYYMMDD format (default: 20250101)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/mawi",
        help="Where to cache downloaded pcap.gz files",
    )
    parser.add_argument(
        "--work-dir",
        default="data/raw/mawi_work",
        help="Where to keep Zeek output logs",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/mawi_extended",
        help="Output directory for extended_v1 NPZ artifacts",
    )
    args = parser.parse_args()

    years = [item.strip() for item in args.years.split(",") if item.strip()]
    days = [item.strip() for item in args.days.split(",") if item.strip()]
    results = build_mawi_dataset(
        years=years,
        days=days,
        raw_dir=args.raw_dir,
        work_dir=args.work_dir,
        out_dir=args.out_dir,
    )
    for item in results:
        print(item)


if __name__ == "__main__":
    main()