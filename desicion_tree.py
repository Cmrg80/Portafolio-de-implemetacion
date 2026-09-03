import numpy as np


class Node:
    """
    Representa un nodo del árbol de decisión.

    Un nodo puede ser:
    - Nodo de decisión: contiene feature y threshold.
    - Hoja: contiene value, que es la clase predicha.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Árbol de decisión implementado desde cero.

    Utiliza Gini Impurity para seleccionar
    las mejores divisiones.
    """

    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        max_features=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        """
        Entrena el árbol con los datos X y las etiquetas y.
        """
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Realiza predicciones para cada fila de X.
        """
        return np.array([
            self._traverse_tree(x, self.root)
            for x in X
        ])

    def _grow_tree(self, X, y, depth=0):
        """
        Construye recursivamente el árbol.
        """

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Condiciones de parada
        if (
            depth >= self.max_depth
            or n_classes == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Selección de características
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            feature_count = min(
                self.max_features,
                n_features
            )

            feature_indices = np.random.choice(
                n_features,
                feature_count,
                replace=False
            )

        # Buscar la mejor división
        best_feature, best_threshold = self._best_split(
            X,
            y,
            feature_indices
        )

        # No existe una división válida
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Dividir los datos
        left_indices, right_indices = self._split(
            X[:, best_feature],
            best_threshold
        )

        # Construir subárbol izquierdo
        left = self._grow_tree(
            X[left_indices],
            y[left_indices],
            depth + 1
        )

        # Construir subárbol derecho
        right = self._grow_tree(
            X[right_indices],
            y[right_indices],
            depth + 1
        )

        # Crear nodo de decisión
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    def _best_split(self, X, y, feature_indices):
        """
        Busca la característica y el umbral
        que producen la menor impureza Gini.
        """

        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature in feature_indices:

            feature_values = X[:, feature]

            thresholds = np.unique(feature_values)

            for threshold in thresholds:

                left_indices, right_indices = self._split(
                    feature_values,
                    threshold
                )

                # Evitar divisiones vacías
                if (
                    len(left_indices) == 0
                    or len(right_indices) == 0
                ):
                    continue

                gini = self._weighted_gini(
                    y[left_indices],
                    y[right_indices]
                )

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split(self, feature_column, threshold):
        """
        Divide los datos en dos grupos:

        izquierda: valor <= threshold
        derecha:   valor > threshold
        """

        left_indices = np.argwhere(
            feature_column <= threshold
        ).flatten()

        right_indices = np.argwhere(
            feature_column > threshold
        ).flatten()

        return left_indices, right_indices

    def _weighted_gini(self, left_y, right_y):
        """
        Calcula la impureza Gini ponderada
        de una división.
        """

        n = len(left_y) + len(right_y)

        weight_left = len(left_y) / n
        weight_right = len(right_y) / n

        gini_left = self._gini(left_y)
        gini_right = self._gini(right_y)

        return (
            weight_left * gini_left
            + weight_right * gini_right
        )

    def _gini(self, y):
        """
        Calcula la impureza Gini de un conjunto.
        """

        if len(y) == 0:
            return 0

        _, counts = np.unique(
            y,
            return_counts=True
        )

        probabilities = counts / len(y)

        return 1 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        """
        Devuelve la clase más frecuente.
        """

        values, counts = np.unique(
            y,
            return_counts=True
        )

        return values[np.argmax(counts)]

    def _traverse_tree(self, x, node):
        """
        Recorre el árbol para obtener
        la predicción de una muestra.
        """

        # Llegamos a una hoja
        if node.is_leaf():
            return node.value

        # Decidir hacia qué hijo avanzar
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(
                x,
                node.left
            )

        return self._traverse_tree(
            x,
            node.right
        )
