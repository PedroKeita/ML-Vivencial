# Classificação e Regressão com Python - Sinais EMG e Aerogeradores

Este projeto foi estabelecido pela UNIFOR, ele implementa métodos de **aprendizado supervisionado e não-supervisionado** aplicados a três problemas distintos:

1. **Regressão Linear**: Previsão de potência de aerogeradores.
2. **Classificação de sinais EMG faciais**: Usando K-NN e Bayes Gaussiano.
3. **Redução de dimensionalidade e clusterização**: Aplicado a imagens faciais com PCA, t-SNE e UMAP, seguido de K-means e K-medoids.

---

## Estrutura de pastas
```
ML-PROJECT/
│
├── classificacao/
│ ├── __init__.py
│ ├── bayes_validator.py
│ ├── kfold.py
│ ├── loading_date.py
│ ├── main.py
│ ├── random_validator.py
│ └── view.py
│
├── data/
│ ├── iRecFac/
│ ├── aerogerador.dat
│ └── EMGDataset.csv
│
├── nao_supervisionado/
│ ├── __init__.py
│ ├── cluster_runner.py
│ ├── dunn_index.py
│ ├── loading_images.py
│ ├── main.py
│ ├── normalize.py
│ ├── pca_runner.py
│ ├── tsne_runner.py
│ └── umap_runner.py
│
├── notebooks/
│ ├──classificao.ipynb
│ ├── nao_supervisionado.ipynb
│ └── regressao.ipynb
│
├── requirements.txt
└── README.md
```

## Instalação do projeto

### Passo 1 - Clonar o repositório:
```
git clone https://github.com/PedroKeita/ML-Vivencial.git
``` 

### Passo 2 - Criar ambiente virtual .venv:
```
python3 -m venv .venv
```

### Passo 3 - Ativar o ambiente virtual:
Linux/macOS:
```
source .venv/bin/activate
```

Windows (cmd):
```
.venv\Scripts\activate
```

Windows (PowerShell):
```
.venv\Scripts\activate.ps1
```

### Passo 4 - Instalar dependências:
```
pip install -r requirements.txt
```

## Executando o projeto
O projeto foi estruturado em módulos e cada um deles pode ser executado indicivudalmente.

### 1. Classificação de sinais EMG
```
python -m classificacao.main
```

### 2. Regressão Linear (Aerogeradores)
```
python -m regressao.main
```

### 3. Aprendizado Não-Supervisionado (imagens de rostos)
```
python -m nao_supervisionado.main
```

## Download do Relatório
[📄 Baixar Relatório](relatorio.pdf)


by: Pedro Lucas Farias de Melo

