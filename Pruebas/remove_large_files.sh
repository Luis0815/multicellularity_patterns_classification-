#!/bin/bash

# Lista de archivos grandes a eliminar
FILES_TO_REMOVE=(
    "morfoespacio_dummy/centroides_tipo/persistencia/distancias/distancias_bottleneck_dim0.csv"
    "morfoespacio_dummy/centroides_tipo/persistencia/distancias/distancias_wasserstein_dim1.csv"
    "morfoespacio_dummy/centroides_tipo/persistencia/distancias/distancias_wasserstein_dim0.csv"
    "morfoespacio_dummy/centroides_tipo/persistencia/distancias/.ipynb_checkpoints/distancias_wasserstein_dim1-checkpoint.csv"
    "morfoespacio_dummy/centroides_tipo/persistencia/distancias/.ipynb_checkpoints/distancias_bottleneck_dim1-checkpoint.csv"
    "morfoespacio_dummy/centroides_tipo/persistencia/distancias/.ipynb_checkpoints/distancias_bottleneck_dim0-checkpoint.csv"
)

# Eliminar cada archivo del historial
for file in "${FILES_TO_REMOVE[@]}"; do
    git filter-repo --invert-paths --path "$file"
done
