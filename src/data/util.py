"""Utility functions for data processing."""

from gzip import GzipFile
from os import makedirs, remove
from os.path import basename, dirname, join, splitext
from tarfile import open as open_tar
from typing import Optional
from urllib.request import urlretrieve
from zipfile import ZipFile

from nibabel import Nifti1Image
from nibabel import save as nib_save
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm  # type: ignore


def download(source_url: str, target_path: str, description: Optional[str] = None) -> None:
    """Download file from source_url to target_path"""
    makedirs(dirname(target_path), exist_ok=True)
    progress_bar = None
    previous_recieved = 0

    def _show_progress(block_num, block_size, total_size):
        nonlocal progress_bar, previous_recieved
        if progress_bar is None:
            progress_bar = tqdm(unit="B", total=total_size)
            if description is not None:
                progress_bar.set_description(description)
        downloaded = block_num * block_size
        if downloaded < total_size:
            progress_bar.update(downloaded - previous_recieved)
            previous_recieved = downloaded
        else:
            progress_bar.close()

    urlretrieve(source_url, target_path, _show_progress)


def untar(file_path: str, target_dir: str | None = None, remove_after: bool = True) -> None:
    """Untar file"""
    if target_dir is None:
        target_dir = dirname(file_path)
    makedirs(target_dir, exist_ok=True)
    tar = open_tar(file_path)
    tar.extractall(target_dir)
    if remove_after:
        remove(file_path)


def unzip(file_path: str, target_dir: str | None = None, remove_after: bool = True) -> None:
    """Unzip file"""
    if target_dir is None:
        target_dir = dirname(file_path)
    makedirs(target_dir, exist_ok=True)
    with ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(target_dir)
    if remove_after:
        remove(file_path)


def ungzip(file_path: str, target_file: str | None = None, remove_after: bool = True) -> None:
    """Ungzip file"""
    if target_file is None:
        target_file = _get_target_path(file_path)
    with GzipFile(file_path, "r") as gzip_file:
        with open(target_file, "wb") as ungzipped_file:
            ungzipped_file.write(gzip_file.read())
    if remove_after:
        remove(file_path)


def save_nifti(data: ndarray | Tensor, path: str, affine: ndarray | Tensor) -> None:
    """Save data to path."""
    if isinstance(data, Tensor):
        data = data.numpy(force=True)
    if isinstance(affine, Tensor):
        affine = affine.numpy(force=True)
    nib_save(
        Nifti1Image(
            data,
            affine=affine,
        ),
        path,
    )


def _get_target_path(file_path: str) -> str:
    return join(dirname(file_path), splitext(basename(file_path))[0])
