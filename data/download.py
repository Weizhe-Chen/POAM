from pathlib import Path
from urllib import request
import numpy as np
from PIL import Image


def download_environment(name):
    path = Path(f"./images/{name.lower()}.jpg")
    if not path.is_file():
        print(f"Downloading to {path}...this step might take some time.")
        request.urlretrieve(
            url="https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
            + f"{name}.SRTMGL1.2.jpg",
            filename=path,
        )
        print("Done")


def preprocess_environment(image_path, array_path, resize=(360, 360)):
    print(f"Preprocessing {image_path}...")
    image = Image.open(image_path).convert("L")
    image = image.resize(size=resize, resample=Image.BICUBIC)
    array = np.array(image).astype(np.float64)
    np.savez_compressed(array_path, array)
    print(f"Saved to {array_path}.")


if __name__ == "__main__":
    name = "N35W107"
    Path(f"./images").mkdir(parents=True, exist_ok=True)
    Path(f"./arrays").mkdir(parents=True, exist_ok=True)
    image_path = Path(f"./images/{name.lower()}.jpg")
    array_path = Path(f"./arrays/{name.lower()}")
    download_environment(name)
    preprocess_environment(image_path, array_path)
