'''
This code for the generation of the synthetic ellipses ground-truth images was developed by Auras et al.[1]

[1] Alexander Auras, Martin Burger, Samira Kabri, Michael Moeller, and Michael Schopf-Kuester. 
    A neural operator view on U-Nets for inverse imaging problems. 2026.
    URL https://arxiv.org/abs/2608.05839
'''

from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm.auto import trange
from typing_extensions import Self, override


class EllipsesDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        image_count: int,
        image_size: tuple[int, int] | int,
        ellipses_per_image: int,
        *,
        binary_output: bool = False,
        min_excentricity: float = 0.975,
        max_excentricity: float = 1.0,
        ellipse_scales: tuple[float, float] = (0.01, 0.06),
        ellipse_intensities: tuple[float, float] = (0.1, 1.0),
        normalize_intensities: bool = True,
        seed: int | None = None,
        smooth: bool = False,
    ) -> None:
        super().__init__()
        self.__image_count = image_count
        self.__image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.__ellipses_per_image = ellipses_per_image
        self.__binary_output = binary_output
        self.__min_excentricity = min_excentricity
        self.__max_excentricity = max_excentricity
        self.__ellipse_scales = ellipse_scales
        self.__ellipse_intensities = ellipse_intensities
        self.__normalize_intensities = normalize_intensities
        self.__seed = seed if seed is not None else int(torch.randint(2**32, (1,)).item())
        self.__smooth = smooth
        if smooth:
            self.__blurrer = v2.GaussianBlur(kernel_size=101, sigma=(100.0))
        self.__file = None

    @classmethod
    def from_file(cls: type[Self], path: str | Path) -> Self:
        instance = __class__.__new__(cls)
        instance.__file = Path(path)
        with h5py.File(path) as hdf_file:
            instance.__image_count = len(hdf_file)
        return instance

    def __len__(self) -> int:
        return self.__image_count

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]: # generate an image with random ellipses based on the index as seed for reproducibility, or load from file if available
        if index < 0:
            raise IndexError()
        if index >= self.__image_count:
            raise StopIteration()
        if self.__file is not None:
            with h5py.File(self.__file) as hdf_file:
                input_ = cast(Any, hdf_file[str(index)])[:]
            return {"input": torch.from_numpy(input_)}
        generator = torch.Generator()
        generator.manual_seed((index + self.__seed) % 2**32)
        sizes = torch.rand((1, 1, self.__ellipses_per_image, 2), generator=generator).sort(dim=-1, descending=True).values
        sizes = self.__ellipse_scales[0] + (self.__ellipse_scales[1] - self.__ellipse_scales[0]) * sizes
        tmp = sizes[..., 1] > (1.0 - self.__min_excentricity**2) ** 0.5 * sizes[..., 0]
        if tmp.any():
            sizes[torch.stack([torch.full_like(tmp, False), tmp], dim=-1)] = (1.0 - self.__min_excentricity**2) ** 0.5 * sizes[tmp][:, 0]
        tmp = sizes[..., 1] < (1.0 - self.__max_excentricity**2) ** 0.5 * sizes[..., 0]
        if tmp.any():
            sizes[torch.stack([torch.full_like(tmp, False), tmp], dim=-1)] = (1.0 - self.__max_excentricity**2) ** 0.5 * sizes[tmp][:, 0]
        positions = torch.rand((1, 1, self.__ellipses_per_image, 2), generator=generator)
        angles = torch.pi * torch.rand((1, 1, self.__ellipses_per_image), generator=generator)
        intensities = self.__ellipse_intensities[0] + (self.__ellipse_intensities[1] - self.__ellipse_intensities[0]) * torch.rand(
            (1, 1, self.__ellipses_per_image), generator=generator
        )
        coords = torch.stack(torch.meshgrid(torch.linspace(0.0, 1.0, self.__image_size[1]), torch.linspace(0.0, 1.0, self.__image_size[0]), indexing="xy"), dim=-1)[:, :, None]
        distances = 1 / sizes[..., 0] * (torch.cos(angles) * (coords[..., 0] - positions[..., 0]) + torch.sin(angles) * (coords[..., 1] - positions[..., 1])) ** 2
        distances += 1 / sizes[..., 1] * (torch.cos(angles) * (coords[..., 1] - positions[..., 1]) - torch.sin(angles) * (coords[..., 0] - positions[..., 0])) ** 2
        tmp = distances <= 1.0
        distances[tmp] = (tmp * intensities)[tmp]
        distances[~tmp] = 0.0
        if self.__binary_output:
            groundtruth = (distances.sum(-1) > 0.0).to(torch.get_default_dtype())
        else:
            groundtruth = distances.sum(-1)
        if self.__normalize_intensities:
            groundtruth = groundtruth / (self.__ellipses_per_image * self.__ellipse_intensities[1])
        if self.__smooth:
            groundtruth = self.__blurrer(groundtruth.unsqueeze(0))[0]
        return {"input": groundtruth[None]}

    def save_to_file(self, path: str | Path, progress: bool = False) -> None: # save dataset to file for faster loading in the future
        if self.__file is not None:
            raise RuntimeError("Dataset already loaded from file")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as hdf_file:
            if progress:
                iter_ = trange(len(cast(Sized, self)))
            else:
                iter_ = range(len(cast(Sized, self)))
            for index in iter_:
                item = self[index]
                hdf_file.create_dataset(str(index), data=item["input"].numpy())