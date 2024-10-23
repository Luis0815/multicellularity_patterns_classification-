# Análisis Topológico de Datos para la Clasificación de Patrones Multicelulares

## Descripción General

Este repositorio contiene el código y los datos utilizados para analizar el patrón espacial de poblaciones celulares heterogéneas a través del **Análisis Topológico de Datos (TDA)**. El objetivo del proyecto es explorar cómo las propiedades variables de adhesión entre células resultan en patrones y estructuras multicelulares distintas. Al aplicar técnicas de TDA, como la **homología persistente**, podemos caracterizar y clasificar estos patrones.

### Objetivos Principales:
1. **Estudiar los efectos de la adhesión célula-célula** en la formación de patrones en sistemas multicelulares.
2. **Clasificar la disposición espacial** de distintos tipos celulares usando características topológicas extraídas a través del TDA..
3. **Aplicar TDA a diversos conjuntos de datos**, incluyendo etapas de progresión del cáncer anogenital, para investigar cómo la disposición celular en estas etapas se correlaciona con la progresión de la enfermedad.

### Metodología:
El pipeline incluye los siguientes pasos:
- Extracción de los centróides de las ubicaciones celulares a partir de imágenes de muestras de tejido.
- Cálculo de **complejos de Rips** y **homología persistente** a partir de estos datos espaciales.
- Análisis de diagramas de persistencia utilizando las distancias de **Bottleneck** y **Wasserstein** para comparar diferentes patrones.
- Visualización de los resultados para interpretar las relaciones espaciales entre tipos celulares en diferentes etapas del desarrollo tumoral.

### Conjuntos de Datos:
- Datos simulados variando la adhesion celular.
- Datos extraídos de imágenes de cáncer incluyendo carcinoma, displasia de alto grado y sus respectivas regiones de estroma adyacente.


### Trabajo en Curso:
- Definir parametros para la clasificacion.
- Exploración de la disposición espacial de las células tumorales.
- Intento de predecir la progresión de la enfermedad en función de los patrones espaciales identificados.

## Bibliografía

1. **Topological Data Analysis of Spatial Patterning in Heterogeneous Cell Populations: Clustering and Sorting with Varying Cell-Cell Adhesion**  
   [Nature Scientific Reports](https://www.nature.com/articles/s41540-023-00302-8)
   
2. **Persistent Homology Based Characterization of the Breast Cancer Immune Microenvironment: A Feasibility Study**  
   [Dagstuhl Reports](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SoCG.2020.11)
   
3. **An Introduction to Topological Data Analysis: Fundamental and Practical Aspects for Data Scientists**  
   [Semantic Scholar](https://www.semanticscholar.org/reader/aff16209e232d38fc94a5b0c72067b88d106453f)
   
4. **Comparison of Persistence Diagrams**  
   [arXiv](https://ar5iv.labs.arxiv.org/html/2003.01352)
