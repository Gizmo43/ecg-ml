import hashlib
from pathlib import Path


def calculate_md5(fpath: str | Path, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    md5 = hashlib.md5(usedforsecurity=False)
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def calculate_sha256(fpath: str | Path, chunk_size: int = 1024 * 1024) -> str:
    sha256 = hashlib.sha256()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_md5(file_path: Path, hash_sum: str) -> bool:
    return calculate_md5(file_path) == hash_sum


def check_integrity(file_path: Path, md5: str) -> bool:
    if not file_path.exists():
        return False
    return check_md5(file_path, md5)
