import numpy as np
from PIL import Image
from glob import glob
import os
from typing import Tuple, List

def loading_images(path: str = "../data/RecFac", image_size: tuple = (128,120), exts: tuple = (".png")) -> Tuple[np.ndarray, List[str], List[str]]:

    files = []
    for ext in exts:
        files += glob(os.path.join(path,"**", f"*{ext}"), recursive = True)
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError(f"Nenhuma imagem foi encontrada em: {path}. Eextensões: {exts}")
    
    X = []
    labels = []
    for f in files:
        image = Image.open(f).convert("L") # aplicar grayscale
        image = image.resize((image_size[0], image_size[1]))
        arr = np.asarray(image, dtype=np.float32)
        X.append(arr.flatten())

        label = os.path.basename(os.path.dirname(f))
        labels.append(label)
    
    return np.vstack(X), files, labels