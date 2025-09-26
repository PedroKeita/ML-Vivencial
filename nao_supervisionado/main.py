from nao_supervisionado import (
    loading_images,
    normalize,
    tsne_runner,
    pca_runner,
    umap_runner,
)

def main():
    print("🔹 Carregando imagens...")

    X, files, labels = loading_images("data/RecFac")
   

    print("🔹 Normalizando dados...")
    X = normalize(X)

    print("🔹 Executando t-SNE...")
    tsne_runner(X, save_dir="plots")

    print("🔹 Executando PCA...")
    pca_runner(X, save_dir="plots")

    print("🔹 Executando UMAP...")
    umap_runner(X, save_dir="plots")

if __name__ == "__main__":
    main()
