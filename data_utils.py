import numpy as np
import pandas as pd

class DatasetMetadata:
    """
    Analiza el conjunto de datos de referencia para registrar tipos de columnas,
    límites de dominio, estadísticas continuas y distribuciones empíricas.
    """
    def __init__(self, df: pd.DataFrame, categorical_cols: list):
        """
        :param df: Pandas DataFrame que sirve como dataset de referencia (background)
        :param categorical_cols: Lista con los nombres de las columnas categóricas
        """
        self.df = df
        self.feature_names = list(df.columns)
        self.categorical_cols = list(categorical_cols)
        self.continuous_cols = [c for c in self.feature_names if c not in self.categorical_cols]
        
        # Mapeamos los índices de las columnas para optimizar accesos vectorizados en NumPy
        self.cat_indices = [self.feature_names.index(col) for col in self.categorical_cols]
        self.cont_indices = [self.feature_names.index(col) for col in self.continuous_cols]
        
        self.m = len(self.feature_names)
        self.h = len(self.categorical_cols)
        
        # Calcular estadísticas para normalizar columnas continuas
        self.std_devs = {}
        self.min_max = {}
        for col in self.continuous_cols:
            std = df[col].std()
            # Si la desviación estándar es 0 (constante), usamos un valor mínimo para evitar división por cero
            self.std_devs[col] = std if std > 0 else 1e-5
            self.min_max[col] = (df[col].min(), df[col].max())
            
        # Almacenar valores únicos para columnas categóricas
        self.categories = {}
        for col in self.categorical_cols:
            self.categories[col] = df[col].unique().tolist()

    def sample_empirical(self, feature_name: str, size: int = 1) -> np.ndarray:
        """
        Muestrea valores de la distribución empírica real de una característica.
        En lugar de usar distribuciones uniformes o gaussianas que generen datos falsos,
        selecciona registros aleatorios directamente del dataset de referencia.
        """
        values = self.df[feature_name].values
        return np.random.choice(values, size=size)


def calculate_distance(x: np.ndarray, Z: np.ndarray, metadata: DatasetMetadata) -> np.ndarray:
    """
    Calcula la distancia híbrida LORE d(x, z) de forma vectorizada.
    
    :param x: Instancia factual original (1D NumPy array de tamaño m)
    :param Z: Matriz de vecindad sintética (2D NumPy array de forma N x m)
    :param metadata: Instancia de DatasetMetadata
    :return: 1D NumPy array de tamaño N con las distancias de x a cada z en Z
    """
    N, m = Z.shape
    
    # 1. Distancia categórica (SimpleMatch)
    if metadata.h > 0:
        # Z[:, indices]!= x[indices] realiza comparación elemento a elemento para N filas
        # El operador!= funciona perfectamente tanto para strings como para enteros en arrays tipo object
        mismatches = Z[:, metadata.cat_indices]!= x[metadata.cat_indices]
        simple_match = np.sum(mismatches, axis=1) / metadata.h
    else:
        simple_match = np.zeros(N)
        
    # 2. Distancia continua (NormEuclid)
    num_continuous = m - metadata.h
    if num_continuous > 0:
        # Convertimos los subconjuntos a float64 para realizar operaciones matemáticas seguras
        Z_cont = Z[:, metadata.cont_indices].astype(np.float64)
        x_cont = x[metadata.cont_indices].astype(np.float64)
        
        # Array con las desviaciones estándar en el orden correcto
        stds = np.array([metadata.std_devs[col] for col in metadata.continuous_cols])
        
        # Diferencia normalizada por desviación estándar
        diff_normalized = (Z_cont - x_cont) / stds
        
        # Distancia Euclidiana Normalizada
        norm_euclid = np.sqrt(np.sum(diff_normalized ** 2, axis=1) / num_continuous)
    else:
        norm_euclid = np.zeros(N)
        
    # 3. Suma ponderada de las distancias según la proporción de variables h/m
    w_cat = metadata.h / m
    w_cont = num_continuous / m
    
    distances = (w_cat * simple_match) + (w_cont * norm_euclid)
    return distances