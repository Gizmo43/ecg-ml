import os
import shutil
from pathlib import Path
from typing import ClassVar

import requests
from tqdm import tqdm

from .hash import calculate_sha256, check_integrity


# TODO:create path dir
class YaDiskDataMixin:
    resources: ClassVar[list[tuple[str, str]]] = []
    name = "Unknown"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

        os.makedirs(self.path, exist_ok=True)

        # if not self.exists():
        #     self.download()

    def exists(self) -> bool:
        return all(
            check_integrity(self.path / file_name, hash_sum)
            for file_name, hash_sum in self.resources
        )

    def download(self) -> None:
        raise RuntimeError("Unimplemented")

    def _yadisk_download(self, public_key: str) -> None:
        url = "https://cloud-api.yandex.net/v1/disk/public/resources"
        params = {"public_key": f"https://disk.yandex.ru/d/{public_key}"}

        resp = requests.get(url, params=params)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Unable connect to {url} with {resp.status_code} status code"
            )

        archive_data = resp.json()
        download_url = archive_data["file"]
        file_size = archive_data["size"]
        file_hash = archive_data["sha256"]

        archive_path = self.path / "archive.zip"

        resp = requests.get(download_url, stream=True)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Unable connect to {url} with {resp.status_code} status code"
            )

        with tqdm(total=file_size, unit="B", unit_scale=True) as progress:
            progress.set_description(f"Downloading dataset '{self.name}'")
            with open(archive_path, "wb") as file:
                for buffer in resp.iter_content(chunk_size=1024 * 1024):
                    file.write(buffer)
                    progress.update(len(buffer))

        if calculate_sha256(archive_path, chunk_size=1024) != file_hash:
            raise RuntimeError  # TODO

        shutil.unpack_archive(archive_path, self.path)

        # Deleting unregistered resources
        resources_filenames = [filename for filename, _ in self.resources]
        for file_path in self.path.iterdir():
            if file_path.name not in resources_filenames:
                os.remove(file_path)
