import numpy as np
import dnn_app_utils_v3 as dnn

def load_system():
    """Carga pesos y parámetros de normalización una sola vez"""
    # 1. Cargar Pesos
    parameters = np.load('mis_pesos_entrenados.npy', allow_pickle=True).item()
    
    # 2. Cargar Normalización (Debes haber guardado esto previamente)
    # Si no los guardaste, ponlos a mano aquí copiándolos de tu print(mins/maxs) del entrenamiento
    # Ejemplo ficticio (REEMPLAZA CON TUS VALORES REALES del print de norm_data):
    # Orden: [Sin, Cos, PacketLoss, ActiveUsers, Latency, Bandwidth]
    # Como Sin/Cos siempre son min -1 y max 1, y PacketLoss 0-1 (si es %/100), ajusta:
    
    # REEMPLAZA ESTO CON TUS ARRAYS REALES 'mins' y 'maxs' DE LAS 6 COLUMNAS DE INPUT
    norm_params = np.load('norm_params.npy', allow_pickle=True).item()
    mins = norm_params['mins'][0:6] # Asegurar que solo tomamos los 6 inputs
    maxs = norm_params['maxs'][0:6]
    
    return parameters, mins, maxs

def preprocess_live_data(hour, packet_loss, active_users, latency, bandwidth, mins, maxs):
    """Prepara un solo dato crudo para la red"""
    
    # 1. Ingeniería de características (Hora -> Sin/Cos)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    # 2. Crear vector (debe tener el mismo orden que tu entrenamiento)
    # [hour_sin, hour_cos, Packet_Loss_Rate, Active_Users, Latency_ms, Bandwidth_MHz]
    raw_vector = np.array([hour_sin, hour_cos, packet_loss, active_users, latency, bandwidth])
    
    # 3. Normalizar (Usando los params del entrenamiento)
    # Fórmula: (x - min) / (max - min)
    input_norm = (raw_vector - mins) / (maxs - mins + 1e-10)
    
    # 4. Reshape para la red (n_x, 1) -> (6, 1)
    input_norm = input_norm.reshape(6, 1)
    
    return input_norm

def predict_throughput(input_norm, parameters):
    """Ejecuta la red"""
    prediction = dnn.predict(input_norm, parameters)
    return prediction[0,0] # Devolver el valor escalar

# --- SIMULACIÓN EN TIEMPO REAL ---

# 1. Inicio del sistema
parameters, mins, maxs = load_system()
print("Sistema de IA 5G cargado. Esperando datos...")

# 2. Llega un dato nuevo (Ejemplo: 2 PM, 0.01 loss, 50 usuarios, 20ms, 100Mhz)
# DATOS CRUDOS (Humanos)
hora_actual = 14 
pkt_loss = 0.05       # 5%
users = 1500          # Usuarios activos
latencia = 15         # ms
ancho_banda = 20      # MHz

# 3. Proceso
input_vector = preprocess_live_data(hora_actual, pkt_loss, users, latencia, ancho_banda, mins, maxs)
pred_norm = predict_throughput(input_vector, parameters)

# 4. Des-normalizar salida (Opcional, si tu Target también estaba normalizado)
# Necesitas el min/max del TARGET (índice 6) para esto.
# target_real = pred_norm * (max_y - min_y) + min_y

print(f"\n--- PREDICCIÓN LIVE ---")
print(f"Input: Hora {hora_actual}:00, Users {users}")
print(f"Throughput Predicho (Normalizado): {pred_norm:.4f}")
# print(f"Throughput Predicho (Mbps): {target_real:.2f} Mbps")
