#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import sys

def calcular_rips_y_persistencia(ruta_centroides):
    # Crear la carpeta para guardar los diagramas de persistencia si no existe
    ruta_persistencia = os.path.join(ruta_centroides, "persistencia")
    if not os.path.exists(ruta_persistencia):
        os.makedirs(ruta_persistencia)

    # Obtener una lista de todos los archivos CSV en la ruta especificada
    archivos_csv = [archivo for archivo in os.listdir(ruta_centroides) if archivo.endswith('.csv')]

    # Iterar sobre cada archivo de centroides
    for archivo_csv in archivos_csv:
        # Construir la ruta completa del archivo
        ruta_completa = os.path.join(ruta_centroides, archivo_csv)
        
        # Leer el archivo CSV
        df = pd.read_csv(ruta_completa)
        
        # Extraer las coordenadas de los centroides
        centroides_x = df['centroides_x'].tolist()
        centroides_y = df['centroides_y'].tolist()
        
        # Convertir las coordenadas de los centroides a un formato adecuado para Gudhi
        puntos = np.array(list(zip(centroides_x, centroides_y)))

        # Calcular el complejo de Rips con un radio específico
        rips_complex = gd.RipsComplex(points=puntos, max_edge_length=40)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

        # Visualizar el complejo de Rips y el diagrama de persistencia
        plt.figure(figsize=(12, 5))

        # Subplot 1: Complejo de Rips
        plt.subplot(1, 2, 1)
        plt.scatter(centroides_x, centroides_y, color='black', label='Centroides')
        for simplex in simplex_tree.get_skeleton(1):  # Obtener las aristas
            if len(simplex[0]) == 2:  # Asegurarse de que solo sean aristas (simplexes de dimensión 1)
                arista = simplex[0]
                x = [centroides_x[i] for i in arista]
                y = [centroides_y[i] for i in arista]
                plt.plot(x, y, color='gray', linestyle='-', linewidth=1)
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.title(f'Complejo de Rips ({archivo_csv})')
        plt.legend()

        # Guardar la imagen del complejo de Rips
        nombre_imagen_rips = f"{os.path.splitext(archivo_csv)[0]}_complejo_rips.png"
        ruta_imagen_rips = os.path.join(ruta_persistencia, nombre_imagen_rips)
        plt.tight_layout()
        plt.savefig(ruta_imagen_rips)
        plt.close()  # Cerrar el gráfico para no sobrecargar la memoria

        # Calcular el diagrama de persistencia
        diag = simplex_tree.persistence()

        # Corregir la estructura de los datos de persistencia para el DataFrame
        persistencia_aplanada = []
        for d in diag:
            dimension, (birth, death) = d
            persistencia_aplanada.append([dimension, birth, death])

        # Guardar los datos de persistencia en un DataFrame
        diagram_df = pd.DataFrame(persistencia_aplanada, columns=['dimension', 'birth', 'death'])

        # Filtrar solo para las dimensiones 0 y 1 (si aplica)
        diagram_df = diagram_df[diagram_df['dimension'] <= 1]

        # Guardar el diagrama de persistencia en formato CSV para futuras comparaciones
        nombre_diagrama_csv = f"{os.path.splitext(archivo_csv)[0]}.csv"
        ruta_diagrama_csv = os.path.join(ruta_persistencia, nombre_diagrama_csv)
        diagram_df.to_csv(ruta_diagrama_csv, index=False)

        # Subplot 2: Diagrama de Persistencia
        plt.figure(figsize=(6, 6))  # Gráfico más pequeño para el diagrama
        gd.plot_persistence_diagram(diag)
        plt.title(f'Diagrama de Persistencia ({archivo_csv})')
        plt.xlabel('Birth')
        plt.ylabel('Death')

        # Guardar la imagen del diagrama de persistencia
        nombre_imagen_persistencia = f"{os.path.splitext(archivo_csv)[0]}_diagrama_persistencia.png"
        ruta_imagen_persistencia = os.path.join(ruta_persistencia, nombre_imagen_persistencia)
        plt.tight_layout()
        plt.savefig(ruta_imagen_persistencia)
        plt.close()  # Cerrar el gráfico para no sobrecargar la memoria

        print(f'Gráfico del Complejo de Rips guardado como {nombre_imagen_rips} y Diagrama de Persistencia guardado como {nombre_imagen_persistencia} en {ruta_persistencia}.')

    return ruta_persistencia

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta_centroides = sys.argv[1]  # Ruta de los centroides
        calcular_rips_y_persistencia(ruta_centroides)
    else:
        print("Por favor, especifica la ruta de los centroides.")
