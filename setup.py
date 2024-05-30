#!/usr/bin/env python3
import io
import os
import re
from typing import List

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        print(version_match.group(1))
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def _load_requirements(
    path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#"
) -> List[str]:
    """Load requirements from a file
    >>> _load_requirements(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


package_name = "src"
readme = open("README.md").read()
version = find_version(package_name, "__init__.py")
install_requires = _load_requirements(os.path.dirname(os.path.realpath(__file__)))

setup(
    name=package_name,
    version=version,
    author="Anonymous",
    author_email="anonymous@anonymous",
    description="Anonymous",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/anonymous",
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": ["isort", "black", "pyright"],
        "test": ["pytest"],
    },
    test_suite="test",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
