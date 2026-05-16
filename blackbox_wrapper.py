import numpy as np

class BlackBoxWrapper:
    """
    Polymorphic Wrapper (Adapter Pattern) que estandariza la interfaz de 
    cualquier modelo predictivo para que LORE interactúe con él de forma agnóstica.
    """
    def __init__(self, model, threshold: float = 0.5):
        """
        :param model: El modelo de caja negra (TensorFlow, Sklearn, PyTorch, etc.)
        :param threshold: Umbral para modelos que retornan probabilidades (ej. TensorFlow Sigmoid)
        """
        self.model = model
        self.threshold = threshold
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """
        Detecta dinámicamente el framework al que pertenece el modelo 
        para evitar acoplamientos rígidos de librerías.
        """
        model_class_str = str(type(self.model)).lower()
        
        if "tensorflow" in model_class_str or "keras" in model_class_str:
            return "tensorflow"
        elif "sklearn" in model_class_str or "xgboost" in model_class_str:
            return "sklearn"
        else:
            return "generic"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Recibe una matriz de características (NumPy array) de forma (N, m) 
        y retorna un array unidimensional de enteros NumPy de forma (N,)
        que representa las clases predichas (0 o 1).
        """
        # Aseguramos que la entrada sea float64 para consistencia matemática
        X_clean = np.asarray(X, dtype=np.float64)

        if self.model_type == "tensorflow":
            # TF.predict por defecto genera barras de progreso. Las silenciamos con verbose=0
            probs = self.model.predict(X_clean, verbose=0)
            
            # Caso 1: Salida de una sola neurona con Sigmoid (Binaria) -> forma (N, 1) o (N,)
            if probs.shape[-1] == 1 or len(probs.shape) == 1:
                return (probs.squeeze() >= self.threshold).astype(int)
            
            # Caso 2: Salida Multiclase con Softmax (ej. dos columnas para binaria o más)
            else:
                return np.argmax(probs, axis=1).astype(int)

        elif self.model_type == "sklearn":
            # Scikit-learn o XGBoost devuelven directamente las etiquetas de clase
            return self.model.predict(X_clean).astype(int)

        else:
            # Caso genérico por si se pasa una función ejecutable (callable) u otro tipo de modelo
            if hasattr(self.model, "predict"):
                return np.array(self.model.predict(X_clean)).astype(int)
            elif callable(self.model):
                return np.array(self.model(X_clean)).astype(int)
            else:
                raise TypeError("El modelo provisto no es ejecutable ni expone un método '.predict()'.")