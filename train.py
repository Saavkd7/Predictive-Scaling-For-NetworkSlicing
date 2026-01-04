import dnn_app_utils_v3 as dnn
import clean_norm as norm
import matplotlib.pyplot as plt
import numpy as np # Necesario para reshape

def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()



file = '5g_slice_traffic.csv'
dataset = norm.norm_data(file)

# --- CORRECCIÓN 1: Selección de Columnas ---
# X son las columnas 0 a 5 (6 columnas)
X = dataset[:, 0:6]  
# y es la columna 6 (Throughput). ¡NO es la 7!
y = dataset[:, 6]    

# --- SPLIT (80/20) ---
split_point = int(len(dataset) * 0.8)

X_train_orig = X[:split_point]
y_train_orig = y[:split_point]

X_test_orig = X[split_point:]
y_test_orig = y[split_point:]

# --- CORRECCIÓN 2: Transponer para Andrew Ng Utils ---
# Pasamos de (8000, 6) a (6, 8000)
X_train = X_train_orig.T
X_test = X_test_orig.T

# IMPORTANTE: y debe ser fila (1, 8000), no vector (8000,)
# Si no haces esto, tendrás errores de broadcasting
y_train = y_train_orig.reshape(1, -1)
y_test = y_test_orig.reshape(1, -1)

print(f"Shape X_train: {X_train.shape}") # Debe decir (7, m)
print(f"Shape y_train: {y_train.shape}") # Debe decir (1, m)
#Verification with data
# print(f"los dies primero de y: {y_train_orig[:10]}")
# print(f"los 10 ultimos de y: {y_test_orig[:-10]}")
#---------------------------------------------------
# # --- VISUALIZACIÓN ---
# plt.close('all')
# plt.figure(figsize=(10,5))
# plt.hist(y_train.flatten(), bins=20) # flatten() para que matplotlib no se queje
# plt.title('Throughput Distribution (Normalized)')
# plt.xlabel("Normalized Demand (0-1)")
# plt.ylabel("Frequency")
# plt.show()
# [Input, Hidden 1, Hidden 2, Hidden 3, Output]
layers_dims = [6, 20, 7, 5, 1] 
parameters=dnn.initialize_parameters_deep(layers_dims)
num_iterations=2500
learning_rate=0.0075
print_cost=False
costs=[] 
for i in range(0,num_iterations):
    AL, caches=dnn.L_model_forward(X_train,parameters)
    cost=dnn.compute_cost(AL,y_train)
    grads=dnn.L_model_backward(AL,y_train,caches)
    parameters=dnn.update_parameters(parameters,grads,learning_rate)
    if i % 100 == 0 or i==num_iterations-1 :
        print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
    if i%100==0:
        costs.append(cost)
plot_costs(costs, learning_rate)


print("--- EVALUACIÓN FINAL ---")

# 1. Predicciones en Train (para ver si aprendió)
print("Resultados en Entrenamiento:")
pred_train = dnn.predict1(X_train, parameters) # Asegúrate de que esta sea la versión de regresión
dnn.evaluate_model(pred_train, y_train)

# 2. Predicciones en Test (LA PRUEBA DE FUEGO)
print("\nResultados en Test (Datos nunca vistos):")
pred_test = dnn.predict1(X_test, parameters)
metrics = dnn.evaluate_model(pred_test, y_test)
#GUARDAR PESOS 
# Guardar
np.save('mis_pesos_entrenados.npy', parameters)
print("Pesos guardados exitosamente.")
# Opcional: Visualizar predicción vs realidad en un pequeño tramo
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten()[:100], label='Realidad (Normalizada)', color='blue')
plt.plot(pred_test.flatten()[:100], label='Predicción AI', color='red', linestyle='dashed')
plt.title("Predicción vs Realidad (Primeras 100 muestras de Test)")
plt.legend()
plt.show()


# ##si quiero cargar despues \
# # Cargar
# parameters = np.load('mis_pesos_entrenados.npy', allow_pickle=True).item()