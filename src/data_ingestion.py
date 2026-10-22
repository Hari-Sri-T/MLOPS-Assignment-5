"""Stage 1 - data_ingestion

Downloads the Cats vs Dogs dataset (filtered version, ~68 MB, 2,000 train
and 1,000 validation images) and extracts it into data/raw/.
The extracted folder is a DVC-tracked output, so `dvc push` uploads it to
the S3 remote and teammates/CI can fetch it with `dvc pull`.
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.utils import ensure_dir, get_logger, load_params

log = get_logger("data_ingestion")


def download(url: str, dest: Path) -> Path:
    if url.startswith("file://"):
        src = Path(url[len("file://"):])
        shutil.copy(src, dest)
    else:
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, dest)
    log.info("Saved archive to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def main() -> None:
    params = load_params()["data"]
    raw_dir = ensure_dir(params["raw_dir"])
    archive = raw_dir / "dataset.zip"

    download(params["source_url"], archive)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()

    dataset_dir = Path(params["dataset_dir"])
    for split in ("train", "validation"):
        for cls in sorted(p.name for p in (dataset_dir / split).iterdir() if p.is_dir()):
            n = len(list((dataset_dir / split / cls).glob("*")))
            log.info("%-10s %-6s %5d images", split, cls, n)


if __name__ == "__main__":
    main()
