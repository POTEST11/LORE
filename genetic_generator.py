import numpy as np
from data_utils import calculate_distance

class GeneticNeighborhoodGenerator:
    """
    Implementa el algoritmo genético ad-hoc de LORE para explorar 
    la frontera de decisión de un modelo de caja negra.
    """
    def __init__(self, population_size=500, generations=10, pc=0.5, pm=0.2):
        """
        :param population_size: N, tamaño de la población.
        :param generations: G, número de generaciones de evolución.
        :param pc: Probabilidad de cruce (crossover).
        :param pm: Probabilidad de mutación.
        """
        self.N = population_size
        self.G = generations
        self.pc = pc
        self.pm = pm

    def _evaluate_fitness(self, x, Z, b_x, b_Z, distances, target_factual):
        """
        Calcula la función de aptitud dual matemática de LORE.
        """
        # Indicadora de identidad: 1 si z es exactamente igual a x, 0 en caso contrario
        I_x_z = np.all(Z == x, axis=1).astype(int)
        
        if target_factual:
            # Recompensa si la predicción de la caja negra es la misma
            I_b = (b_Z == b_x).astype(int)
        else:
            # Recompensa si la predicción de la caja negra es diferente
            I_b = (b_Z!= b_x).astype(int)
            
        # Fórmula de aptitud
        fitness = I_b + (1 - distances) - I_x_z
        return fitness

    def _crossover(self, P):
        """Aplica Two-Point Crossover a la población."""
        N, m = P.shape
        P_next = P.copy()
        
        for i in range(0, N - 1, 2):
            if np.random.rand() < self.pc and m >= 2:
                # Seleccionar dos puntos de cruce al azar
                pt1, pt2 = np.sort(np.random.choice(m, 2, replace=False))
                
                # Intercambiar genes (características) entre padres
                temp = P_next[i, pt1:pt2].copy()
                P_next[i, pt1:pt2] = P_next[i+1, pt1:pt2]
                P_next[i+1, pt1:pt2] = temp
                
        return P_next

    def _mutate(self, P, metadata):
        """Mutación basada en la distribución empírica de los datos de referencia."""
        N, m = P.shape
        for i in range(N):
            if np.random.rand() < self.pm:
                # Elegir un índice de característica al azar para mutar
                feat_idx = np.random.randint(m)
                feat_name = metadata.feature_names[feat_idx]
                
                # Muestrear un valor realista desde el dataset de referencia
                new_val = metadata.sample_empirical(feat_name, size=1).item()
                P[i, feat_idx] = new_val
                
        return P

    def generate(self, x, wrapper, metadata, target_factual=True):
        """
        Ejecuta el bucle de evolución genética.
        Si target_factual=True, genera Z_(=). Si False, genera Z_(≠).
        """
        m = len(x)
        
        # 1. Inicialización: La población P_0 se llena con copias exactas de x
        P = np.tile(x, (self.N, 1))
        
        # Obtenemos la predicción base para la instancia original
        b_x = wrapper.predict(x.reshape(1, -1))

        for gen in range(self.G):
            # Obtener predicciones y distancias de la generación actual
            b_P = wrapper.predict(P)
            distances = calculate_distance(x, P, metadata)
            
            # Evaluar aptitud
            fitness = self._evaluate_fitness(x, P, b_x, b_P, distances, target_factual)
            
            # Selección por truncamiento: seleccionamos la mitad superior más apta
            best_indices = np.argsort(fitness)[::-1][:self.N // 2]
            P_selected = P[best_indices]
            
            # Duplicamos a los mejores para restaurar el tamaño de la población N
            P_next = np.vstack((P_selected, P_selected)) 
            
            # Aplicar operadores genéticos
            P_crossed = self._crossover(P_next)
            P_mutated = self._mutate(P_crossed, metadata)
            
            P = P_mutated

        # Evaluación final al terminar las G generaciones
        b_P = wrapper.predict(P)
        distances = calculate_distance(x, P, metadata)
        fitness = self._evaluate_fitness(x, P, b_x, b_P, distances, target_factual)
        
        # Ordenamos y retornamos la población final según su puntuación de aptitud
        final_indices = np.argsort(fitness)[::-1]
        return P[final_indices]