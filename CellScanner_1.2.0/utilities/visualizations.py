import numpy as np
import umap


class UmapVisualization:
    """
    UmapVisualization class is used to create UMAP embeddings of the dataframe that are used in low-dimensional
    projection of the results as well as the part of the training dataset.
    """
    def __init__(self, data: np.ndarray, num_cores: int, dims: int) -> None:
        """
        Args:
            data (np.array): dataframe.
            num_cores: amount of CPU cores used for the computation of embeddings
        """
        self.data = data
        self.num_cores = num_cores
        self.dims = dims
        self.embeddings = self.__reduction()

    def __reduction(self) -> np.ndarray:
        """
        Performs dimensional reduction to either 2D or 3D as specified in the settings.
        Returns:
            fitted (np.array): calculated embeddings.
        """
        reducer = umap.UMAP(n_components=self.dims, n_neighbors=25, min_dist=0.1, metric="euclidean",
                            n_jobs=self.num_cores)
        fitted = reducer.fit_transform(self.data)
        return fitted

    def get_embeddings(self) -> np.ndarray:
        """
        Returns:
            embeddings (np.array): calculated embeddings.
        """
        return self.embeddings


class MplVisualization:
    pass


class TrainingVisualization:
    # Ideas: custom callback to draw matplotlib plot or plotly (preferably plotly) after each epoch. Otherwise, consider
    # using a tensorboard callback.
    pass
