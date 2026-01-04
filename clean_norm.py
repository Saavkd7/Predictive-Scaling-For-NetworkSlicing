import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def norm_data(file):
    data = pd.read_csv(file, delimiter=',')
    
    # 1. Ingeniería de características (Ciclo Horario)
    data['hour_sin'] = np.sin(2 * np.pi * data['Hour_of_Day'] / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['Hour_of_Day'] / 24.0)
    
    # 2. Selección de columnas (ORDEN CRÍTICO)
    # Metemos TODO lo que sea numérico aquí, incluido Packet Loss.
    # Orden deseado: [Inputs... , Target]
    features_order = [
        'hour_sin', 
        'hour_cos', 
        'Packet_Loss_Rate',  # <--- METELO AQUÍ Y OLVÍDATE DE PROBLEMAS
        'Active_Users', 
        'Latency_ms', 
        'Bandwidth_MHz', 
        'Throughput_Demand_Mbps' # Target al final (índice 6)
    ]
    
    dataset = data[features_order].values
    
    # 3. Normalización Global
    # Así garantizas que NADA se escape del rango 0-1
    maxs = dataset.max(axis=0)
    mins = dataset.min(axis=0)
    
    # Evitamos división por cero si una columna es constante (max == min)
    dataset_norm = (dataset - mins) / (maxs - mins + 1e-10)
    
    # 4. Verificación
    print(f"Shape final: {dataset_norm.shape}") # Debería ser (Filas, 7)
    print(pd.DataFrame(dataset_norm).head())
    
    # Graficar el Target (Índice 6)
    plt.hist(dataset_norm[:, 6]) 
    plt.title('Throughput (Normalized)')
    plt.show()
    
    return dataset_norm
