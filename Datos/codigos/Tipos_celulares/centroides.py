#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def calcular_y_guardar_centroides_separados(ruta_directorio):
    # Crear carpeta para guardar los centroides si no existe
    carpeta_centroides = os.path.join(ruta_directorio, "centroides_tipo")
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
        
        # Inicializar diccionarios para almacenar los centroides por tipo
        centroides_por_tipo = {'A': {'x': [], 'y': [], 'id': []},
                               'B': {'x': [], 'y': [], 'id': []}}

        # Calcular el centroide y el tipo para cada grupo
        for (id_grupo, tipo_grupo), grupo in grupos:
            X_grupo = grupo['X']
            Y_grupo = grupo['Y']
            centroid_x_grupo = X_grupo.mean()
            centroid_y_grupo = Y_grupo.mean()
            
            # Guardar los centroides en la lista correspondiente al tipo
            centroides_por_tipo[tipo_grupo]['x'].append(centroid_x_grupo)
            centroides_por_tipo[tipo_grupo]['y'].append(centroid_y_grupo)
            centroides_por_tipo[tipo_grupo]['id'].append(id_grupo)  # Guardar el ID del grupo
        
        # Guardar los centroides de cada tipo en archivos CSV separados
        for tipo, datos in centroides_por_tipo.items():
            if datos['x']:  # Si hay datos para ese tipo
                df_centroides = pd.DataFrame({"centroides_x": datos['x'], "centroides_y": datos['y'], "id": datos['id']})
                
                # Guardar el DataFrame como CSV en la carpeta de centroides
                nombre_archivo_centroides = os.path.splitext(archivo_csv)[0] + f"_{tipo}.csv"
                ruta_archivo_centroides = os.path.join(carpeta_centroides, nombre_archivo_centroides)
                df_centroides.to_csv(ruta_archivo_centroides, index=False)

        # Crear el gráfico de dispersión con centroides rojos y azules
        for tipo in ['A', 'B']:
            color = 'b' if tipo == 'A' else 'r'  # Rojo para tipo 'A', azul para tipo 'B'
            centroides_x_tipo = centroides_por_tipo[tipo]['x']
            centroides_y_tipo = centroides_por_tipo[tipo]['y']
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 centroides.py <ruta_directorio>")
        sys.exit(1)

    ruta_directorio = sys.argv[1]
    calcular_y_guardar_centroides_separados(ruta_directorio)
