import numpy as np
from collections import deque
import tensorflow as tf

class OnlinePredictor:
    def __init__(self, model, buffer_size=50, change_threshold=0.1):
        self.model = model
        # Buffer para guardar los últimos N datos (Ventana deslizante)
        self.memory_buffer = deque(maxlen=buffer_size) 
        
        self.last_prediction = 0.0
        self.change_threshold = change_threshold # El "umbral de no cambio"
        
    def predict_stabilized(self, new_input):
        """
        Realiza la predicción y aplica el filtro de umbral.
        """
        # 1. Obtener la predicción "cruda" del modelo actual
        # Asumimos que new_input tiene la forma (1, features)
        raw_pred = self.model.predict(new_input, verbose=0)[0][0]
        
        # 2. Lógica del Umbral (Hysteresis / Deadband)
        # Solo actualizamos si el cambio es significativo
        diff = abs(raw_pred - self.last_prediction)
        
        if diff > self.change_threshold:
            self.last_prediction = raw_pred
            print(f"--> Cambio detectado ({diff:.3f}). Nueva salida: {raw_pred:.3f}")
        else:
            # Mantenemos el valor anterior para evitar ruido
            # (Opcional: podrías hacer un promedio ponderado si prefieres suavizar)
            pass 
            
        return self.last_prediction

    def update_weights(self, x_real, y_real):
        """
        Agrega el dato real al buffer y re-entrena los pesos.
        """
        # 1. Guardar en memoria (FIFO)
        self.memory_buffer.append((x_real, y_real))
        
        # 2. Solo entrenar si tenemos suficientes datos en la ventana
        if len(self.memory_buffer) >= 10: # Mínimo para un mini-batch
            # Preparamos el batch desde la memoria
            X_batch = np.array([m[0] for m in self.memory_buffer])
            y_batch = np.array([m[1] for m in self.memory_buffer])
            
            # X_batch necesita tener la forma correcta (batch_size, features)
            # Ajusta dimensiones si es necesario: X_batch = X_batch.reshape(...)
            
            # 3. Actualización de pesos (Online Learning)
            # train_on_batch es más rápido que fit() para esto
            loss = self.model.train_on_batch(X_batch, y_batch)
            return loss
        return None

# --- Simulación de uso ---

# Supongamos que ya tienes tu 'model' compilado de Keras
predictor = OnlinePredictor(model=mi_modelo_dnn, buffer_size=100, change_threshold=0.2)

# Bucle de tiempo real (pseudo-código)
while recibiendo_datos:
    # 1. Llega el input del sensor
    nuevo_sensor_data = recibir_sensor() 
    
    # 2. Predecimos (usando los pesos actuales)
    output_control = predictor.predict_stabilized(nuevo_sensor_data)
    actuar_en_sistema(output_control)
    
    # 3. Unos milisegundos después, obtenemos la "Realidad" o usamos la del paso anterior
    # para entrenar. Es crucial NO entrenar con el dato que acabamos de intentar predecir
    # hasta que sepamos la respuesta correcta (ground truth).
    valor_real = leer_realidad() 
    
    predictor.update_weights(nuevo_sensor_data, valor_real)
