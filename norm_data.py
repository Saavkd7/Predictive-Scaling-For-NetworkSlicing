import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def norm_data(file):
    data=pd.read_csv(file,delimiter=',')
    data.head()
    # Calculamos el componente X (Seno)
    data['hour_sin'] = np.sin(2 * np.pi * data['Hour_of_Day'] / 24.0)
    # Calculamos el componente Y (Coseno)
    data['hour_cos'] = np.cos(2 * np.pi * data['Hour_of_Day'] / 24.0)
    #Validation
    is_correct = np.allclose(data['hour_sin']**2 + data['hour_cos']**2, 1.0)
    print(f"¿Normalización válida?: {is_correct}")
    data.drop(columns='Hour_of_Day', inplace=True)
    #Moviendo a la primera y segunda posicion respectivamente
    data.insert(0,'hour_sin',data.pop('hour_sin'))
    data.insert(1,'hour_cos',data.pop('hour_cos'))
    data.head()
    #Extracting data to an array to normalize 
    dataset = data[['Active_Users', 'Latency_ms', 'Bandwidth_MHz', 'Throughput_Demand_Mbps']].values
    print(dataset.shape)
    #active_user_max=dataset[:,0].max()
    #NORMALIZATION 
    maxs = (dataset.max(axis=0))
    mins = (dataset.min(axis=0))
    dataset = (dataset - mins) / (maxs - mins + 1e-10)
    print(pd.DataFrame(dataset).head(10))
    data_F = np.column_stack((data['hour_sin'], data['hour_cos'], data['Packet_Loss_Rate'], dataset))
    print(pd.DataFrame(data_F).head(10))
    plt.hist(data_F[:,6])
    plt.title('Throuhput')
    return data_F
