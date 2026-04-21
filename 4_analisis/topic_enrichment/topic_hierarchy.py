'''
topic_hierarchy.py
------------------
Extrae y representa la estructura jerárquica de los clusters
a partir del linkage producido por el clustering jerárquico
(AgglomerativeClustering en scikit-learn / scipy).

El objetivo es pasar de la estructura de árbol implícita en el
dendrograma a una representación explícita y navegable como un
diccionario anidado o un DataFrame de relaciones padre-hijo.
Esto permite construir visualizaciones de jerarquía e interpretar
cómo se agrupan los tópicos a diferentes niveles de granularidad.

Lógica general:
    1. Reconstruir la matriz de linkage desde los vectores del cluster
       (si no se tiene el linkage original) usando scipy
    2. Recorrer el árbol de linkage en orden y construir el árbol de nodos
    3. Asignar a cada nodo hoja el cluster_id correspondiente
    4. Para nodos internos, calcular el cluster dominante (el más frecuente
       entre sus hojas) y la distancia de fusión
    5. Retornar la jerarquía como diccionario anidado y como DataFrame
       de relaciones padre-hijo para facilitar visualización

Funciones públicas:
    build_linkage_matrix          -- construye la matriz de linkage desde vectores
    extract_hierarchy_from_linkage -- extrae árbol padre-hijo desde la matriz de linkage
    assign_cluster_to_nodes       -- asigna cluster dominante a cada nodo interno
    hierarchy_to_dataframe        -- convierte la jerarquía a DataFrame tabular
    get_cluster_depth             -- calcula la profundidad de cada cluster en el árbol
'''

import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


def build_linkage_matrix(
    X: np.ndarray,
    method: str = 'ward',
    metric: str = 'euclidean',
) -> np.ndarray:
    '''
    Construye la matriz de linkage desde la matriz de vectores X
    usando el método y métrica especificados.

    La matriz de linkage tiene shape (n_docs - 1, 4) donde cada fila
    representa una fusión: [índice_izq, índice_der, distancia, n_elementos].

    Parámetros:
        X      -- matriz de vectores (n_docs, n_dims)
        method -- método de enlace: 'ward', 'complete', 'average', 'single'
        metric -- métrica de distancia (para method='ward' solo 'euclidean')

    Retorna:
        Matriz de linkage de scipy (n_docs - 1, 4)
    '''
    logger.info(
        'Construyendo linkage: method=%s, metric=%s, shape=%s',
        method, metric, X.shape
    )

    # Ward solo funciona con distancia euclidiana
    if method == 'ward' and metric != 'euclidean':
        logger.warning(
            'Ward linkage requiere métrica euclidiana. '
            'Cambiando metric de "%s" a "euclidean"', metric
        )
        metric = 'euclidean'

    # Calcular matriz de distancias condensada y luego linkage
    distancias = pdist(X, metric=metric)
    Z = linkage(distancias, method=method)

    logger.info('Linkage construido: %d fusiones para %d documentos', len(Z), len(X))
    return Z


def extract_hierarchy_from_linkage(
    Z: np.ndarray,
    n_docs: int,
) -> dict:
    '''
    Extrae la estructura de árbol completa desde la matriz de linkage.
    Usa scipy.cluster.hierarchy.to_tree para convertir la matriz Z
    en un objeto ClusterNode navegable.

    Parámetros:
        Z      -- matriz de linkage (output de build_linkage_matrix o scipy)
        n_docs -- número total de documentos (hojas del árbol)

    Retorna:
        Diccionario con la representación del árbol:
        {
            'nodo_id': int,
            'es_hoja': bool,
            'indice_doc': int | None,    # solo si es hoja
            'distancia_fusion': float,   # distancia al fusionarse con el padre
            'n_elementos': int,
            'izquierdo': dict | None,
            'derecho': dict | None,
        }
    '''
    raiz = to_tree(Z, rd=False)

    def _recorrer_nodo(nodo) -> dict:
        '''Recorre recursivamente el árbol y construye el diccionario.'''
        es_hoja = nodo.is_leaf()
        resultado = {
            'nodo_id'         : nodo.id,
            'es_hoja'         : es_hoja,
            'indice_doc'      : nodo.id if es_hoja else None,
            'distancia_fusion': round(float(nodo.dist), 6),
            'n_elementos'     : nodo.count,
            'izquierdo'       : None,
            'derecho'         : None,
        }

        if not es_hoja:
            resultado['izquierdo'] = _recorrer_nodo(nodo.left)
            resultado['derecho']   = _recorrer_nodo(nodo.right)

        return resultado

    arbol = _recorrer_nodo(raiz)
    logger.info(
        'Árbol extraído: %d elementos totales, distancia raíz=%.4f',
        arbol['n_elementos'], arbol['distancia_fusion']
    )
    return arbol


def assign_cluster_to_nodes(
    arbol: dict,
    labels: list[int] | np.ndarray,
) -> dict:
    '''
    Recorre el árbol y asigna a cada nodo interno el cluster dominante
    entre todas sus hojas. Las hojas ya tienen su índice de documento
    y se mapean directamente a la etiqueta de cluster correspondiente.

    Parámetros:
        arbol  -- árbol de jerarquía (output de extract_hierarchy_from_linkage)
        labels -- etiquetas de cluster por documento (alineadas por índice)

    Retorna:
        El mismo árbol enriquecido con el campo 'cluster_dominante' en cada nodo
    '''
    labels_array = np.array(labels)

    def _asignar_cluster(nodo: dict) -> list[int]:
        '''
        Retorna la lista de clusters de todas las hojas del nodo.
        Asigna cluster_dominante al nodo como efecto secundario.
        '''
        if nodo['es_hoja']:
            idx = nodo['indice_doc']
            if idx < len(labels_array):
                cluster = int(labels_array[idx])
            else:
                cluster = -1
            nodo['cluster_dominante'] = cluster
            return [cluster]

        # Nodo interno: combinar clusters de ambos hijos
        clusters_izq = _asignar_cluster(nodo['izquierdo'])
        clusters_der = _asignar_cluster(nodo['derecho'])
        todos_clusters = clusters_izq + clusters_der

        # El cluster dominante es el más frecuente (excluyendo ruido si es posible)
        clusters_validos = [c for c in todos_clusters if c != -1]
        if clusters_validos:
            cluster_dominante = Counter(clusters_validos).most_common(1)[0][0]
        else:
            cluster_dominante = -1

        nodo['cluster_dominante'] = cluster_dominante
        return todos_clusters

    _asignar_cluster(arbol)
    logger.info('Clusters asignados a todos los nodos del árbol')
    return arbol


def hierarchy_to_dataframe(
    arbol: dict,
    labels: list[int] | np.ndarray | None = None,
) -> pd.DataFrame:
    '''
    Convierte la jerarquía de árbol a un DataFrame plano de relaciones
    padre-hijo, más fácil de exportar y usar en visualizaciones.

    Cada fila representa un nodo del árbol con información de su padre.

    Parámetros:
        arbol  -- árbol de jerarquía (puede estar enriquecido o no con clusters)
        labels -- etiquetas de cluster opcionales para enriquecer si no están ya

    Retorna:
        DataFrame con columnas:
            nodo_id, padre_id, es_hoja, indice_doc, cluster_dominante,
            distancia_fusion, n_elementos
    '''
    filas = []

    def _recorrer_para_df(nodo: dict, padre_id: int | None) -> None:
        fila = {
            'nodo_id'           : nodo['nodo_id'],
            'padre_id'          : padre_id,
            'es_hoja'           : nodo['es_hoja'],
            'indice_doc'        : nodo.get('indice_doc'),
            'cluster_dominante' : nodo.get('cluster_dominante'),
            'distancia_fusion'  : nodo['distancia_fusion'],
            'n_elementos'       : nodo['n_elementos'],
        }
        filas.append(fila)

        if not nodo['es_hoja']:
            _recorrer_para_df(nodo['izquierdo'], nodo['nodo_id'])
            _recorrer_para_df(nodo['derecho'],   nodo['nodo_id'])

    _recorrer_para_df(arbol, padre_id=None)

    df = pd.DataFrame(filas)
    logger.info(
        'DataFrame de jerarquía generado: %d nodos (%d hojas)',
        len(df), int(df['es_hoja'].sum())
    )
    return df


def get_cluster_depth(
    arbol: dict,
) -> dict[int, int]:
    '''
    Calcula la profundidad de cada nodo en el árbol jerárquico.
    La raíz tiene profundidad 0, sus hijos directos profundidad 1, etc.

    Útil para entender la granularidad de los clusters: clusters que
    se separan a mayor profundidad son más distintos entre sí.

    Parámetros:
        arbol -- árbol de jerarquía (output de extract_hierarchy_from_linkage)

    Retorna:
        Diccionario {nodo_id: profundidad}
    '''
    profundidades: dict[int, int] = {}

    def _recorrer(nodo: dict, profundidad: int) -> None:
        profundidades[nodo['nodo_id']] = profundidad
        if not nodo['es_hoja']:
            _recorrer(nodo['izquierdo'], profundidad + 1)
            _recorrer(nodo['derecho'],   profundidad + 1)

    _recorrer(arbol, profundidad=0)
    logger.info(
        'Profundidades calculadas: max=%d, nodos=%d',
        max(profundidades.values()) if profundidades else 0,
        len(profundidades)
    )
    return profundidades


def build_full_hierarchy(
    X: np.ndarray,
    labels: list[int] | np.ndarray,
    method: str = 'ward',
    metric: str = 'euclidean',
) -> tuple[dict, pd.DataFrame]:
    '''
    Función de conveniencia que ejecuta todo el pipeline de jerarquía
    en un solo paso: linkage -> árbol -> asignación de clusters -> DataFrame.

    Parámetros:
        X      -- matriz de vectores (n_docs, n_dims)
        labels -- etiquetas de cluster por documento
        method -- método de enlace para el linkage
        metric -- métrica de distancia para el linkage

    Retorna:
        Tupla (arbol_enriquecido, dataframe_jerarquia)
    '''
    Z = build_linkage_matrix(X, method, metric)
    arbol = extract_hierarchy_from_linkage(Z, n_docs=len(X))
    arbol = assign_cluster_to_nodes(arbol, labels)
    df_jerarquia = hierarchy_to_dataframe(arbol)

    return arbol, df_jerarquia