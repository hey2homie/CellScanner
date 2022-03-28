import numpy
import pandas as pd
import umap
import matplotlib.pyplot as plt
import data_preparation
from sklearn.mixture import BayesianGaussianMixture


class UnsupervisedClustering:

    def __init__(self, data: pd.DataFrame, cpu: int = 4) -> None:
        self.labels = data["Species"]
        self.data = data.drop("Species", axis=1)
        # TODO: Remove labels and drop column
        self.embeddings = self.__reduction(cpu)
        self.labels_clustering = self.__clustering()

    def __reduction(self, cpu: int = 4) -> numpy.ndarray:
        reducer = umap.UMAP(n_components=3, n_neighbors=25, min_dist=0.1, metric="euclidean", n_jobs=cpu)
        fitted = reducer.fit_transform(self.data)
        return fitted

    def __clustering(self) -> numpy.ndarray:
        # TODO: Adjustable number of clusters through BIC or AIC
        clusters = BayesianGaussianMixture(n_components=2).fit_predict(self.embeddings)
        return clusters

    def visualization(self) -> plt.Figure:
        # TODO: Remove plot with the labels coming from species
        # TODO: Add legend and title
        # TODO: Figure parameters like size and dpi
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2, 1, 1, projection="3d")
        ax.scatter(self.embeddings[:, 0], self.embeddings[:, 1], self.embeddings[:, 2], c=pd.factorize(self.labels)[0])
        ax = fig.add_subplot(2, 1, 2, projection="3d")
        ax.scatter(self.embeddings[:, 0], self.embeddings[:, 1], self.embeddings[:, 2], c=self.labels_clustering)
        return fig

    def save_visuals(self):
        # TODO: Where to save, file name
        self.visualization().savefig("1.png")

    def save_data(self):
        pass


if __name__ == '__main__':
    prepare = data_preparation.DataPreparation()
    prepare.add_files("../references/Blautia_hydrogenotrophica-211119-GAM-1.fcs")
    prepare.add_files("../references/Escherichia_coli-050520-mGAM-1.fcs")
    dataset = prepare.get_object()
    clustering = UnsupervisedClustering(dataset)
    clustering.save_visuals()

