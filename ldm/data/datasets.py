import glob
import io
import os
import random
import time
from typing import Tuple
import duckdb
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
import webdataset as wds
from fsspec import filesystem
from fsspec.implementations.http import HTTPFileSystem
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm.auto import tqdm


def load_image(image: Image.Image, size: int) -> Tensor:
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    if hasattr(TF, "to_image"):
        image = TF.to_image(image)
        image = TF.to_dtype(image, torch.float32, True)
    else:
        image = TF.to_image_tensor(image)
        image = TF.convert_image_dtype(image, torch.float32)
    return image


class ImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        **kwargs,
    ):
        self.root = root
        self.image_size = image_size
        self.paths = np.asarray(
            list(self.find(root)),
        )
        self._last_image = None

    def find(self, root: str) -> str:
        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    yield os.path.join(dirname, filename)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        path = self.paths[idx]
        image = self._last_image
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            pass
        if image is None:
            return torch.zeros(3, self.image_size, self.image_size)
        self._last_image = image
        image = load_image(image, self.image_size)
        return image


class Laion2B(IterableDataset):
    def __init__(self, root: str, image_size: int = 256, **kwargs):
        self.root = root
        self.image_size = image_size
        torch.manual_seed(time.time())
        random.seed(time.time())
        self.data = (
            wds.WebDataset(
                glob.glob(os.path.join(root, "*.tar")),
                wds.ignore_and_continue,
                shardshuffle=True,
            )
            .shuffle(1000)
            .decode("pil")
            .map_dict(jpg=lambda x: load_image(x, self.image_size))
        )

    def __len__(self) -> int:
        return 200000000

    def __iter__(self):
        for data in self.data:
            if "jpg" in data:
                jpg = data["jpg"]

                yield jpg


class Laion400M(Dataset):
    def __init__(self, image_size: int = 512, cache_dir: str | None = None):
        self.image_size = image_size
        self.cache_dir = cache_dir
        self.size = 413871335
        laion400m = duckdb.read_parquet(
            "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-*.parquet",
            hive_partitioning=True,
        )
        laion400m = laion400m.create_view("laion400m").query(
            "laion400m", "SELECT SAMPLE_ID, URL FROM laion400m"
        )
        self._laion400m = laion400m
        self._https: HTTPFileSystem = filesystem("https")

    def __len__(self):
        return self.size

    def __iter__(self):
        return self._image_iter()

    def _id_url_iter(self) -> str:
        duckdb.register_filesystem(self._https)
        while (res := self._laion400m.fetchone()) is not None:
            yield res

    def _download_image(self, url: str) -> Image.Image | None:
        resp = self._https.cat(url, on_error="omit")
        if resp is not None:
            buff = io.BytesIO(resp)
            buff.seek(0)
            image = Image.open(buff).convert("RGB")
            return image

    def _image_iter(self) -> Image.Image:
        for id, url in self._id_url_iter():
            image = self._download_image(url)
            if image is not None:
                image = load_image(image, self.image_size)
                yield image
