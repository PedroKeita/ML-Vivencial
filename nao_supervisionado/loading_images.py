"""
Módulo para carregamento de imagens

Permite carregar imagens de um diretório, convertê-las para grayscale, redimensionar e gerar arrays
prontos para análise.
"""

import numpy as np
from PIL import Image
from glob import glob
import os
from typing import Tuple, List

def loading_images(path: str = "../data/RecFac", image_size: tuple = (128,120), exts: tuple = (".png")) -> Tuple[np.ndarray, List[str], List[str]]:

    """
    Carrega imagens de um diretório, aplica grayscale e redimensionamento.

    Args:
        path (str): Diretório onde estão as imagens.
        image_size (tuple): Tamanho final das imagens (altura, largura).
        exts (tuple): Extensões de arquivos a serem carregados.

    Returns:
        Tuple[np.ndarray, List[str], List[str]]: 
            - X: Array 2D com cada linha representando uma imagem flatten.
            - files: Lista de caminhos completos para os arquivos carregados.
            - labels: Lista de rótulos obtidos a partir do nome da pasta de cada imagem.
    """
    
    files = []
    for ext in exts:
        files += glob(os.path.join(path,"**", f"*{ext}"), recursive = True)
    files = [f for f in files if os.path.isfile(f)]
    if len(files) == 0:
        raise FileNotFoundError(f"Nenhuma imagem foi encontrada em: {path}. Eextensões: {exts}")
    
    X = []
    labels = []
    for f in files:
        image = Image.open(f).convert("L") 
        image = image.resize((image_size[0], image_size[1]))
        arr = np.asarray(image, dtype=np.float32)
        X.append(arr.flatten())

        label = os.path.basename(os.path.dirname(f))
        labels.append(label)
    
    return np.vstack(X), files, labels