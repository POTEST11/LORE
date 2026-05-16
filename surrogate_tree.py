import numpy as np
from sklearn.tree import DecisionTreeClassifier

class SurrogateTreeEvaluator:
    """
    Entrena un árbol de decisión local sobre el vecindario sintético
    y extrae las reglas factuales y contrafactuales navegando por sus nodos.
    """
    def __init__(self, metadata, min_samples_leaf=3, max_depth=None):
        self.metadata = metadata
        # Ajustamos hiperparámetros para evitar sobreajuste extremo en el vecindario local
        self.clf = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf, 
            max_depth=max_depth,
            random_state=42
        )
        self.feature_names = metadata.feature_names

    def fit_local_tree(self, Z, y_Z):
        """Ajusta el modelo subrogado c a los datos locales Z."""
        self.clf.fit(Z, y_Z)
        return self.clf

    def _extract_path(self, node_index):
        """
        Método interno para rastrear el camino desde la raíz hasta un nodo específico.
        Retorna una lista de diccionarios con las condiciones (splits).
        """
        tree_ = self.clf.tree_
        
        def recurse(current_node, current_path=None):
            if current_path is None:
                current_path = []
            if current_node == node_index:
                return current_path
            
            left_child = tree_.children_left[current_node]
            right_child = tree_.children_right[current_node]
            
            if left_child!= -1:
                res = recurse(left_child, current_path + [(current_node, '<=')])
                if res: return res
            if right_child!= -1:
                res = recurse(right_child, current_path + [(current_node, '>')])
                if res: return res
            return None

        raw_path = recurse(0)
        conditions = []
        
        if raw_path is None: 
            return conditions
        
        for node, op in raw_path:
            feat_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            feat_name = self.feature_names[feat_idx]
            
            conditions.append({
                'feature': feat_name, 
                'index': feat_idx, 
                'op': op, 
                'threshold': threshold
            })
            
        return conditions

    def extract_factual_rule(self, x):
        """
        Encuentra el camino (la regla) que explica la predicción de la instancia original x.
        """
        x_reshaped = x.reshape(1, -1).astype(np.float64)
        # Identificar la hoja en la que cae la instancia x
        leaf_id = int(self.clf.apply(x_reshaped)[0])
        
        prediction = self.clf.classes_[np.argmax(self.clf.tree_.value[leaf_id])]
        conditions = self._extract_path(leaf_id)
        
        return {'conditions': conditions, 'prediction': prediction}

    def extract_counterfactual_rules(self, x, factual_prediction):
        """
        Busca caminos que lleven a una predicción distinta y minimicen el número de 
        condiciones falsificadas (métrica nf).
        """
        tree_ = self.clf.tree_
        n_nodes = tree_.node_count
        
        # Identificar todas las hojas del árbol
        leaves = [i for i in range(n_nodes) if tree_.children_left[i] == -1]
        cf_candidates = []
        
        for leaf in leaves:
            pred_class = self.clf.classes_[np.argmax(tree_.value[leaf])]
            
            # Solo nos interesan los nodos que cambian la predicción
            if pred_class!= factual_prediction:
                path_conditions = self._extract_path(leaf)
                
                # Calcular nf: ¿Cuántas condiciones de este camino NO cumple x?
                nf = 0
                for cond in path_conditions:
                    val = float(x[cond['index']])
                    if cond['op'] == '<=' and val > cond['threshold']:
                        nf += 1
                    elif cond['op'] == '>' and val <= cond['threshold']:
                        nf += 1
                        
                cf_candidates.append({
                    'conditions': path_conditions,
                    'prediction': pred_class,
                    'nf': nf
                })
        
        if not cf_candidates:
            return []
            
        # Filtrar seleccionando únicamente aquellos con el costo nf mínimo
        min_nf = min(c['nf'] for c in cf_candidates)
        best_cfs = [c for c in cf_candidates if c['nf'] == min_nf]
        
        return best_cfs