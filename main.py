import pandas as pd
import numpy as np

from decision_tree import DecisionTree


# ==========================================
# 1. CARGAR DATASET
# ==========================================

DATA_PATH = "data/breats-cancer.csv"

df = pd.read_csv(DATA_PATH)

print("==========================================")
print("     RANDOM FOREST FROM SCRATCH")
print("     PRUEBA DEL DECISION TREE")
print("==========================================\n")

print("Dataset cargado correctamente.")
print(f"Registros: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}\n")


# ==========================================
# 2. PREPROCESAMIENTO
# ==========================================

# Eliminar identificador
df = df.drop(columns=["id"])

# Convertir diagnóstico:
# M = 1 (maligno)
# B = 0 (benigno)

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# Separar características y variable objetivo
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values

print("Características utilizadas:", X.shape[1])
print("Variable objetivo: diagnosis")
print("M = 1 (Maligno)")
print("B = 0 (Benigno)\n")


# ==========================================
# 3. DIVISIÓN TRAIN / TEST
# ==========================================

np.random.seed(42)

indices = np.random.permutation(len(X))

train_size = int(0.80 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

print("División del dataset:")
print(f"Entrenamiento: {len(X_train)} registros")
print(f"Prueba:        {len(X_test)} registros\n")


# ==========================================
# 4. CREAR Y ENTRENAR ÁRBOL
# ==========================================

tree = DecisionTree(
    max_depth=5,
    min_samples_split=2,
    max_features=5
)

print("Entrenando Decision Tree...")
tree.fit(X_train, y_train)

print("Entrenamiento terminado.\n")


# ==========================================
# 5. REALIZAR PREDICCIONES
# ==========================================

y_pred = tree.predict(X_test)

print("Predicciones realizadas.\n")


# ==========================================
# 6. MATRIZ DE CONFUSIÓN
# ==========================================

true_positive = np.sum(
    (y_test == 1) & (y_pred == 1)
)

true_negative = np.sum(
    (y_test == 0) & (y_pred == 0)
)

false_positive = np.sum(
    (y_test == 0) & (y_pred == 1)
)

false_negative = np.sum(
    (y_test == 1) & (y_pred == 0)
)


print("==========================================")
print("         MATRIZ DE CONFUSIÓN")
print("==========================================\n")

print("                 Predicción")
print("                 0       1")
print(f"Real 0          {true_negative:3d}     {false_positive:3d}")
print(f"Real 1          {false_negative:3d}     {true_positive:3d}")
print()


# ==========================================
# 7. MÉTRICAS
# ==========================================

total = len(y_test)

accuracy = (
    (true_positive + true_negative)
    / total
)

precision = (
    true_positive
    / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive
    / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1_score = (
    2 * (precision * recall)
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


print("==========================================")
print("              MÉTRICAS")
print("==========================================\n")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
