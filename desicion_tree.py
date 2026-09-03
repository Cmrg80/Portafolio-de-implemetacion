import numpy as np


class Node:
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
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Condiciones para detener el crecimiento
        if (
            depth >= self.max_depth
            or n_classes == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Selección aleatoria de características
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = np.random.choice(
                n_features,
                self.max_features,
                replace=False
            )

        # Buscar la mejor división
        best_feature, best_threshold = self._best_split(
            X,
            y,
            feature_indices
        )

        # Si no encontramos una división válida
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Crear subconjuntos
        left_indices, right_indices = self._split(
            X[:, best_feature],
            best_threshold
        )

        # Crear hijos
        left = self._grow_tree(
            X[left_indices],
            y[left_indices],
            depth + 1
        )

        right = self._grow_tree(
            X[right_indices],
            y[right_indices],
            depth + 1
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    def _best_split(self, X, y, feature_indices):

        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature in feature_indices:

            values = X[:, feature]
            thresholds = np.unique(values)

            for threshold in thresholds:

                left_indices, right_indices = self._split(
                    values,
                    threshold
                )

                if len(left_indices) == 0 or len(right_indices) == 0:
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

        left_indices = np.argwhere(
            feature_column <= threshold
        ).flatten()

        right_indices = np.argwhere(
            feature_column > threshold
        ).flatten()

        return left_indices, right_indices

    def _weighted_gini(self, left_y, right_y):

        n = len(left_y) + len(right_y)

        weight_left = len(left_y) / n
        weight_right = len(right_y) / n

        return (
            weight_left * self._gini(left_y)
            + weight_right * self._gini(right_y)
        )

    def _gini(self, y):

        classes, counts = np.unique(
            y,
            return_counts=True
        )

        probabilities = counts / len(y)

        return 1 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):

        values, counts = np.unique(
            y,
            return_counts=True
        )

        return values[np.argmax(counts)]

    def _traverse_tree(self, x, node):

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
