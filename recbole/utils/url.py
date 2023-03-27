"""
recbole.utils.url
################################
Reference code:
    https://github.com/snap-stanford/ogb/blob/master/ogb/utils/url.py
"""

import urllib.request as ur
import zipfile
import os
import os.path as osp
import errno
from logging import getLogger

from tqdm import tqdm

GBFACTOR = float(1 << 30)


def decide_download(url):
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"]) / GBFACTOR

    ### confirm if larger than 1GB
    if size > 1:
        return (
            input(
                "This will download %.2fGB. Will you proceed? (y/N)\n" % (size)
            ).lower()
            == "y"
        )
    else:
        return True


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder):
    """Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
    """

    filename = url.rpartition("/")[2]
    path = osp.join(folder, filename)
    logger = getLogger()

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        logger.info(f"Using exist file {filename}")
        return path

    logger.info(f"Downloading {url}")

    makedirs(folder)
    data = ur.urlopen(url)

    size = int(data.info()["Content-Length"])

    chunk_size = 1024 * 1024
    num_iter = int(size / chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, "wb") as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description(
                    "Downloaded {:.2f} GB".format(float(downloaded_size) / GBFACTOR)
                )
                f.write(chunk)
    except:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError("Stopped downloading due to interruption.")

    return path


def extract_zip(path, folder):
    """Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
    """
    logger = getLogger()
    logger.info(f"Extracting {path}")
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)


def rename_atomic_files(folder, old_name, new_name):
    """Rename all atomic files in a given folder.

    Args:
        folder (string): The folder.
        old_name (string): Old name for atomic files.
        new_name (string): New name for atomic files.
    """
    files = os.listdir(folder)
    for f in files:
        base, suf = os.path.splitext(f)
        if not old_name in base:
            continue
        if suf not in {".inter", ".user", ".item"}:
            logger = getLogger()
            logger.warning(f"Moving downloaded file with suffix [{suf}].")
        os.rename(
            os.path.join(folder, f),
            os.path.join(folder, base.replace(old_name, new_name) + suf),
        )


if __name__ == "__main__":
    pass
