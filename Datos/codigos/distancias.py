#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import gudhi as gd
import gudhi.wasserstein as gw
import sys

def calcular_distancias(ruta_directorio):
    # Crear la carpeta de salida
    carpeta_salida = os.path.join(ruta_directorio, "distancias")
    os.makedirs(carpeta_salida, exist_ok=True)

    # Almacenar los diagramas de persistencia
    diagramas_persistencia = {}
    diagramas_persistencia_0 = {}

    # Listar todos los archivos CSV en el directorio
    archivos_csv = [f for f in os.listdir(ruta_directorio) if f.endswith('.csv')]

    # Iterar sobre cada archivo CSV
    for archivo_csv in archivos_csv:
        try:
            # Construir la ruta completa del archivo
            ruta_completa = os.path.join(ruta_directorio, archivo_csv)
            
            # Leer el archivo CSV y asignar nombres de columnas
            df = pd.read_csv(ruta_completa, header=0)
            
            # Validar que las columnas necesarias existen
            if 'birth' not in df.columns or 'death' not in df.columns:
                print(f"El archivo {archivo_csv} no contiene las columnas 'birth' o 'death'.")
                continue
            
            # Convertir columnas a flotantes
            df['birth'] = df['birth'].astype(float)
            df['death'] = df['death'].astype(float)

            # Almacenar diagramas de persistencia para dimensiones 0 y 1
            diagramas_persistencia_0[archivo_csv] = df[df['dimension'] == 0][['birth', 'death']].to_numpy()
            diagramas_persistencia[archivo_csv] = df[df['dimension'] == 1][['birth', 'death']].to_numpy()
        
        except Exception as e:
            print(f"Error al procesar el archivo {archivo_csv}: {e}")

    # Calcular distancias entre todos los pares de diagramas
    archivos = list(diagramas_persistencia.keys())
    n = len(archivos)

    # Inicializar matrices de distancias
    distancias_bottleneck_dim1 = np.zeros((n, n))
    distancias_bottleneck_dim0 = np.zeros((n, n))
    distancias_wasserstein_dim1 = np.zeros((n, n))
    distancias_wasserstein_dim0 = np.zeros((n, n))

    # Tolerancia para considerar distancias efectivamente cero
    tolerancia = 1e-10

    for i in range(n):
        for j in range(i, n):
            archivo_i = archivos[i]
            archivo_j = archivos[j]
            diag_i_1 = diagramas_persistencia[archivo_i]
            diag_j_1 = diagramas_persistencia[archivo_j]
            diag_i_0 = diagramas_persistencia_0[archivo_i]
            diag_j_0 = diagramas_persistencia_0[archivo_j]
            
            # Calcular distancias de Bottleneck
            bottleneck_1 = gd.bottleneck_distance(diag_i_1, diag_j_1)
            bottleneck_0 = gd.bottleneck_distance(diag_i_0, diag_j_0)
            
            # Calcular distancias de Wasserstein
            wasserstein_1 = gw.wasserstein_distance(diag_i_1, diag_j_1, order=1)
            wasserstein_0 = gw.wasserstein_distance(diag_i_0, diag_j_0, order=1)
            
            # Almacenar las distancias
            distancias_bottleneck_dim1[i, j] = distancias_bottleneck_dim1[j, i] = bottleneck_1
            distancias_bottleneck_dim0[i, j] = distancias_bottleneck_dim0[j, i] = bottleneck_0
            distancias_wasserstein_dim1[i, j] = distancias_wasserstein_dim1[j, i] = wasserstein_1
            distancias_wasserstein_dim0[i, j] = distancias_wasserstein_dim0[j, i] = wasserstein_0
            
            # Validar que las distancias para el mismo archivo son cero
            if i == j:
                if bottleneck_1 < tolerancia and wasserstein_1 < tolerancia:
                    print(f"Distancias (Dimensión 1) para {archivo_i} son efectivamente cero.")
                else:
                    print(f"Distancias (Dimensión 1) para {archivo_i} no son cero.")
                if bottleneck_0 < tolerancia and wasserstein_0 < tolerancia:
                    print(f"Distancias (Dimensión 0) para {archivo_i} son efectivamente cero.")
                else:
                    print(f"Distancias (Dimensión 0) para {archivo_i} no son cero.")

    # Guardar matrices de distancias en CSV
    pd.DataFrame(distancias_bottleneck_dim1, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_bottleneck_dim1.csv'))
    pd.DataFrame(distancias_bottleneck_dim0, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_bottleneck_dim0.csv'))
    pd.DataFrame(distancias_wasserstein_dim1, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_wasserstein_dim1.csv'))
    pd.DataFrame(distancias_wasserstein_dim0, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_wasserstein_dim0.csv'))

    print(f"Distancias guardadas en {carpeta_salida}.")
    return carpeta_salida

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta_directorio = sys.argv[1]
        calcular_distancias(ruta_directorio)
    else:
        print("Por favor, especifica la ruta del directorio que contiene los diagramas de persistencia.")

