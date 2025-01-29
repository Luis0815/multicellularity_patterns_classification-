#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import Image     # Import the Image function from IPython.display to display images in Jupyter environments.
from os import chdir                  # Import chdir from os module to change the current working directory.
from scipy.spatial import Delaunay  # Importar Delaunay
import numpy as np                    # Import numpy library for working with n-dimensional arrays and mathematical operations.
import gudhi as gd                    # Import gudhi library for computational topology and computational geometry.
import matplotlib.pyplot as plt       # Import pyplot from matplotlib for creating visualizations and graphs.
import argparse                       # Import argparse, a standard library for writing user-friendly command-line interfaces.
import seaborn as sns                 # Import seaborn for data visualization; it's based on matplotl.
import requests # Import requests library to make HTTP requests in Python easily.
import pandas as pd
from scipy.spatial.distance import pdist, squareform 
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import pandas as pd
import os


# In[3]:


#Me ayuda a ver que contiene cada archivo
def mostrar_contenido_csv(ruta_directorio):
    archivos_csv = [archivo for archivo in os.listdir(ruta_directorio) if archivo.endswith('.csv')]
    for archivo_csv in archivos_csv:
        ruta_completa = os.path.join(ruta_directorio, archivo_csv)
        df = pd.read_csv(ruta_completa, header=None)
        print(f'Contenido de {archivo_csv}:')
        print(df)
        print("\n" + "="*50 + "\n")


# In[4]:


def calcular_y_guardar_centroides(ruta_directorio):
    # Crear carpeta para guardar los centroides si no existe
    carpeta_centroides = os.path.join(ruta_directorio, "centroides")
    if not os.path.exists(carpeta_centroides):
        os.makedirs(carpeta_centroides)

    # Obtener una lista de todos los archivos CSV en la ruta especificada
    archivos_csv = [archivo for archivo in os.listdir(ruta_directorio) if archivo.endswith('.csv')]

    # Iterar sobre cada archivo CSV
    for archivo_csv in archivos_csv:
        # Construir la ruta completa del archivo
        ruta_completa = os.path.join(ruta_directorio, archivo_csv)
        
        # Leer el archivo CSV y asignar nombres de columnas
        df = pd.read_csv(ruta_completa, header=None)
        nombres_columnas = ['id', 'Tipo', 'X', 'Y', 'Z']  # Reemplazar con los nombres de columnas apropiados
        df.columns = nombres_columnas
        
        # Agrupar los datos por el ID y el Tipo
        grupos = df.groupby(['id', 'Tipo'])
        
        # Inicializar listas para almacenar los centroides y tipos
        centroides_x = []
        centroides_y = []
        tipos_centroides = []
        
        # Calcular el centroide y el tipo para cada grupo
        for (id_grupo, tipo_grupo), grupo in grupos:
            X_grupo = grupo['X']
            Y_grupo = grupo['Y']
            centroid_x_grupo = X_grupo.mean()
            centroid_y_grupo = Y_grupo.mean()
            centroides_x.append(centroid_x_grupo)
            centroides_y.append(centroid_y_grupo)
            tipos_centroides.append(tipo_grupo)  # Guardar el tipo del grupo
        
        # Crear un DataFrame para almacenar los centroides
        df_centroides = pd.DataFrame({"centroides_x": centroides_x, "centroides_y": centroides_y, "id": tipos_centroides})
        
        # Guardar el DataFrame como CSV en la carpeta de centroides
        nombre_archivo_centroides = os.path.splitext(archivo_csv)[0] + ".csv"
        ruta_archivo_centroides = os.path.join(carpeta_centroides, nombre_archivo_centroides)
        df_centroides.to_csv(ruta_archivo_centroides, index=False)
        
        # Crear el gráfico de dispersión con centroides rojos y azules
        for tipo in set(tipos_centroides):
            color = 'b' if tipo == 'A' else 'r'  # Rojo para tipo 'A', azul para tipo 'B'
            indices_tipo = [i for i, t in enumerate(tipos_centroides) if t == tipo]
            centroides_x_tipo = [centroides_x[i] for i in indices_tipo]
            centroides_y_tipo = [centroides_y[i] for i in indices_tipo]
            plt.scatter(centroides_x_tipo, centroides_y_tipo, color=color, label=f'Tipo {tipo}')
        
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.title(f'Centroides por Tipo ({archivo_csv})')
        plt.legend()
        
        # Guardar la imagen del gráfico en la carpeta de centroides
        ruta_imagen = os.path.join(carpeta_centroides, os.path.splitext(archivo_csv)[0] + "_centroides.png")
        plt.savefig(ruta_imagen)
        plt.close()  # Cerrar el gráfico para no sobrecargar la memoria

        print(f"Centroides y gráfico guardados para {archivo_csv} en {carpeta_centroides}.")

    return carpeta_centroides 


# In[5]:


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
        rips_complex = gd.RipsComplex(points=puntos, max_edge_length=8)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

        # Visualizar el complejo de Rips y el diagrama de persistencia
        plt.figure(figsize=(24, 10))

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


# In[7]:


def calcular_distancia_bottleneck(ruta_directorio):
    # Crear la carpeta de salida
    carpeta_salida = os.path.join(ruta_directorio, "distancia de cuello de botella")
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
            df = pd.read_csv(ruta_completa, header=0)  # Usar la primera fila como cabecera
            
            # Asegúrate de que el DataFrame tenga las columnas correctas
            if 'birth' not in df.columns or 'death' not in df.columns:
                print(f"El archivo {archivo_csv} no contiene las columnas 'birth' o 'death'.")
                continue
            
            # Convertir columnas 'birth' y 'death' a flotantes
            df['birth'] = df['birth'].astype(float)
            df['death'] = df['death'].astype(float)

            # Almacenar el diagrama de persistencia en dimensión 1
            diag_1 = df[df['dimension'] == 1][['birth', 'death']].to_numpy()
            diagramas_persistencia[archivo_csv] = diag_1
            
            # Almacenar el diagrama de persistencia en dimensión 0
            diag_0 = df[df['dimension'] == 0][['birth', 'death']].to_numpy()
            diagramas_persistencia_0[archivo_csv] = diag_0
            
        except Exception as e:
            print(f"Error al procesar el archivo {archivo_csv}: {e}")

    # Calcular las distancias de Bottleneck entre todos los pares de diagramas de persistencia
    archivos = list(diagramas_persistencia.keys())
    n = len(archivos)
    tolerancia = 1e-10  # Tolerancia para considerar distancias efectivamente como cero

    # Inicializar matrices de distancias
    distancias_dim1 = np.zeros((n, n))
    distancias_dim0 = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            archivo_i = archivos[i]
            archivo_j = archivos[j]
            diag_i = diagramas_persistencia[archivo_i]
            diag_j = diagramas_persistencia[archivo_j]
            diag_i_0 = diagramas_persistencia_0[archivo_i]
            diag_j_0 = diagramas_persistencia_0[archivo_j]
            
            # Calcular distancias de Bottleneck
            distancia_bottleneck = gd.bottleneck_distance(diag_i, diag_j)
            distancia_bottleneck_0 = gd.bottleneck_distance(diag_i_0, diag_j_0)
            
            distancias_dim1[i, j] = distancia_bottleneck
            distancias_dim1[j, i] = distancia_bottleneck
            distancias_dim0[i, j] = distancia_bottleneck_0
            distancias_dim0[j, i] = distancia_bottleneck_0
            
            # Verificar que la distancia de Bottleneck para el mismo archivo es efectivamente cero
            if i == j:
                if distancia_bottleneck < tolerancia:
                    print(f"Distancia de Bottleneck (Dimensión 1) para {archivo_i} es efectivamente cero.")
                else:
                    print(f"Distancia de Bottleneck (Dimensión 1) para {archivo_i} no es cero: {distancia_bottleneck}")
                
                if distancia_bottleneck_0 < tolerancia:
                    print(f"Distancia de Bottleneck (Dimensión 0) para {archivo_i} es efectivamente cero.")
                else:
                    print(f"Distancia de Bottleneck (Dimensión 0) para {archivo_i} no es cero: {distancia_bottleneck_0}")

    # Guardar las matrices de distancias en CSV
    pd.DataFrame(distancias_dim1, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_bottleneck_dim1.csv'))
    pd.DataFrame(distancias_dim0, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_bottleneck_dim0.csv'))


# In[9]:


def calcular_distancia_wasserstein(ruta_directorio):
    # Crear la carpeta de salida
    carpeta_salida = os.path.join(ruta_directorio, "distancia de wasserstein")
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
            df = pd.read_csv(ruta_completa, header=0)  # Usar la primera fila como cabecera
            
            # Asegúrate de que el DataFrame tenga las columnas correctas
            if 'birth' not in df.columns or 'death' not in df.columns:
                print(f"El archivo {archivo_csv} no contiene las columnas 'birth' o 'death'.")
                continue
            
            # Convertir columnas 'birth' y 'death' a flotantes
            df['birth'] = df['birth'].astype(float)
            df['death'] = df['death'].astype(float)

            # Almacenar el diagrama de persistencia en dimensión 1
            diag_1 = df[df['dimension'] == 1][['birth', 'death']].to_numpy()
            diagramas_persistencia[archivo_csv] = diag_1
            
            # Almacenar el diagrama de persistencia en dimensión 0
            diag_0 = df[df['dimension'] == 0][['birth', 'death']].to_numpy()
            diagramas_persistencia_0[archivo_csv] = diag_0
            
        except Exception as e:
            print(f"Error al procesar el archivo {archivo_csv}: {e}")

    # Calcular las distancias de Wasserstein entre todos los pares de diagramas de persistencia
    archivos = list(diagramas_persistencia.keys())
    n = len(archivos)
    tolerancia = 1e-10  # Tolerancia para considerar distancias efectivamente como cero

    # Inicializar matrices de distancias
    distancias_dim1 = np.zeros((n, n))
    distancias_dim0 = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            archivo_i = archivos[i]
            archivo_j = archivos[j]
            diag_i = diagramas_persistencia[archivo_i]
            diag_j = diagramas_persistencia[archivo_j]
            diag_i_0 = diagramas_persistencia_0[archivo_i]
            diag_j_0 = diagramas_persistencia_0[archivo_j]
            
            # Calcular distancias de Wasserstein
            distancia_wasserstein = gw.wasserstein_distance(diag_i, diag_j, order=1)
            distancia_wasserstein_0 = gw.wasserstein_distance(diag_i_0, diag_j_0, order=1)
            
            distancias_dim1[i, j] = distancia_wasserstein
            distancias_dim1[j, i] = distancia_wasserstein
            distancias_dim0[i, j] = distancia_wasserstein_0
            distancias_dim0[j, i] = distancia_wasserstein_0
            
            # Verificar que la distancia de Wasserstein para el mismo archivo es efectivamente cero
            if i == j:
                if distancia_wasserstein < tolerancia:
                    print(f"Distancia de Wasserstein (Dimensión 1) para {archivo_i} es efectivamente cero.")
                else:
                    print(f"Distancia de Wasserstein (Dimensión 1) para {archivo_i} no es cero: {distancia_wasserstein}")
                
                if distancia_wasserstein_0 < tolerancia:
                    print(f"Distancia de Wasserstein (Dimensión 0) para {archivo_i} es efectivamente cero.")
                else:
                    print(f"Distancia de Wasserstein (Dimensión 0) para {archivo_i} no es cero: {distancia_wasserstein_0}")

    # Guardar las matrices de distancias en CSV
    pd.DataFrame(distancias_dim1, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_wasserstein_dim1.csv'))
    pd.DataFrame(distancias_dim0, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_wasserstein_dim0.csv'))


# In[10]:


def distancias(ruta_directorio):
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
            df = pd.read_csv(ruta_completa, header=0)  # Usar la primera fila como cabecera
            
            # Asegúrate de que el DataFrame tenga las columnas correctas
            if 'birth' not in df.columns or 'death' not in df.columns:
                print(f"El archivo {archivo_csv} no contiene las columnas 'birth' o 'death'.")
                continue
            
            # Convertir columnas 'birth' y 'death' a flotantes
            df['birth'] = df['birth'].astype(float)
            df['death'] = df['death'].astype(float)

            # Almacenar el diagrama de persistencia en dimensión 1
            diag_1 = df[df['dimension'] == 1][['birth', 'death']].to_numpy()
            diagramas_persistencia[archivo_csv] = diag_1
            
            # Almacenar el diagrama de persistencia en dimensión 0
            diag_0 = df[df['dimension'] == 0][['birth', 'death']].to_numpy()
            diagramas_persistencia_0[archivo_csv] = diag_0
            
        except Exception as e:
            print(f"Error al procesar el archivo {archivo_csv}: {e}")

    # Calcular las distancias entre todos los pares de diagramas de persistencia
    archivos = list(diagramas_persistencia.keys())
    n = len(archivos)
    tolerancia = 1e-10  # Tolerancia para considerar distancias efectivamente como cero

    # Inicializar matrices de distancias
    distancias_bottleneck_dim1 = np.zeros((n, n))
    distancias_bottleneck_dim0 = np.zeros((n, n))
    distancias_wasserstein_dim1 = np.zeros((n, n))
    distancias_wasserstein_dim0 = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            archivo_i = archivos[i]
            archivo_j = archivos[j]
            diag_i = diagramas_persistencia[archivo_i]
            diag_j = diagramas_persistencia[archivo_j]
            diag_i_0 = diagramas_persistencia_0[archivo_i]
            diag_j_0 = diagramas_persistencia_0[archivo_j]
            
            # Calcular distancias de Bottleneck
            distancia_bottleneck = gd.bottleneck_distance(diag_i, diag_j)
            distancia_bottleneck_0 = gd.bottleneck_distance(diag_i_0, diag_j_0)
            
            # Calcular distancias de Wasserstein
            distancia_wasserstein = gw.wasserstein_distance(diag_i, diag_j, order=1)
            distancia_wasserstein_0 = gw.wasserstein_distance(diag_i_0, diag_j_0, order=1)
            
            # Almacenar las distancias en las matrices
            distancias_bottleneck_dim1[i, j] = distancias_bottleneck_dim1[j, i] = distancia_bottleneck
            distancias_bottleneck_dim0[i, j] = distancias_bottleneck_dim0[j, i] = distancia_bottleneck_0
            distancias_wasserstein_dim1[i, j] = distancias_wasserstein_dim1[j, i] = distancia_wasserstein
            distancias_wasserstein_dim0[i, j] = distancias_wasserstein_dim0[j, i] = distancia_wasserstein_0
            
            # Verificar que las distancias para el mismo archivo son efectivamente cero
            if i == j:
                if distancia_bottleneck < tolerancia and distancia_wasserstein < tolerancia:
                    print(f"Distancias (Dimensión 1) para {archivo_i} son efectivamente cero.")
                else:
                    print(f"Distancias (Dimensión 1) para {archivo_i} no son cero: Bottleneck = {distancia_bottleneck}, Wasserstein = {distancia_wasserstein}")
                
                if distancia_bottleneck_0 < tolerancia and distancia_wasserstein_0 < tolerancia:
                    print(f"Distancias (Dimensión 0) para {archivo_i} son efectivamente cero.")
                else:
                    print(f"Distancias (Dimensión 0) para {archivo_i} no son cero: Bottleneck = {distancia_bottleneck_0}, Wasserstein = {distancia_wasserstein_0}")

    # Guardar las matrices de distancias en CSV
    pd.DataFrame(distancias_bottleneck_dim1, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_bottleneck_dim1.csv'))
    pd.DataFrame(distancias_bottleneck_dim0, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_bottleneck_dim0.csv'))
    pd.DataFrame(distancias_wasserstein_dim1, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_wasserstein_dim1.csv'))
    pd.DataFrame(distancias_wasserstein_dim0, index=archivos, columns=archivos).to_csv(os.path.join(carpeta_salida, 'distancias_wasserstein_dim0.csv'))

    print("Los cálculos de distancias se han completado y guardado en la carpeta 'distancias'.")

    return carpeta_salida 


# In[11]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def extract_levels(filename):
    parts = filename.replace('.csv', '').split('_')
    return [int(parts[0]), int(parts[1]), int(parts[2])]

def sort_filenames(filenames):
    return sorted(filenames, key=extract_levels)

def reorder_matrix(matrix, filenames):
    sorted_filenames = sort_filenames(filenames)
    indices = [filenames.index(name) for name in sorted_filenames]
    reordered_matrix = matrix[np.ix_(indices, indices)]
    return reordered_matrix, sorted_filenames

def plot_heatmap_and_clustermap(matrix, filenames, title, output_dir):
    reordered_matrix, sorted_filenames = reorder_matrix(matrix, filenames)
    
    # Heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(reordered_matrix, 
                xticklabels=sorted_filenames, 
                yticklabels=sorted_filenames, 
                cmap='viridis', 
                annot=False,
                fmt=".2f", 
                cbar_kws={'label': 'Distancia'})
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(f"Heatmap de {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{title}.png'))
    plt.close()

    # Clustermap
    clustermap = sns.clustermap(reordered_matrix, 
                                xticklabels=sorted_filenames, 
                                yticklabels=sorted_filenames, 
                                cmap='viridis', 
                                annot=False,
                                fmt=".2f", 
                                figsize=(15, 12),
                                dendrogram_ratio=(.1, .2),
                                cbar_pos=(0, .2, .03, .4),
                                cbar_kws={'label': 'Distancia'})
    clustermap.ax_heatmap.set_xticklabels(
        clustermap.ax_heatmap.get_xticklabels(), 
        rotation=90, 
        fontsize=6)
    clustermap.ax_heatmap.set_yticklabels(
        clustermap.ax_heatmap.get_yticklabels(), 
        rotation=0, 
        fontsize=6)
    plt.title(f"Clustermap de {title}", pad=80)
    plt.savefig(os.path.join(output_dir, f'clustermap_{title}.png'))
    plt.close()

def crear_visualizaciones(ruta_directorio):
    carpeta_visualizacion = os.path.join(ruta_directorio, "visualizacion")
    os.makedirs(carpeta_visualizacion, exist_ok=True)

    archivos_csv = [f for f in os.listdir(ruta_directorio) if f.endswith('.csv')]

    for archivo in archivos_csv:
        nombre_base = os.path.splitext(archivo)[0]
        
        distancias = pd.read_csv(os.path.join(ruta_directorio, archivo), index_col=0)
        
        # Extraer los nombres de los archivos de las columnas
        filenames = list(distancias.columns)
        
        plot_heatmap_and_clustermap(distancias.values, filenames, nombre_base, carpeta_visualizacion)

    print("Las visualizaciones se han  guardado.")


# In[ ]:




