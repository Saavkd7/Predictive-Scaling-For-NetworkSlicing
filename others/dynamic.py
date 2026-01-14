import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# MÓDULO 1: SIMULACIÓN DE DATOS (Igual que antes)
# ==========================================
class NetworkEnvironment:
    def __init__(self, n_samples=2000, drift_start=1000):
        self.n_samples = n_samples
        self.drift_start = drift_start
    
    def generate_traffic(self):
        np.random.seed(42)
        X = np.random.normal(0.3, 0.05, (1, self.n_samples)) # (Features, Examples)
        Y = X * 2 + np.random.normal(0, 0.05, (1, self.n_samples))
        
        # Evento Drift
        X[:, self.drift_start:] = np.random.normal(0.8, 0.1, (1, self.n_samples - self.drift_start))
        Y[:, self.drift_start:] = (X[:, self.drift_start:] * 4) + 0.5 
        return X, Y

# ==========================================
# MÓDULO 2: EL CEREBRO REAL (DEEP NEURAL NETWORK)
# ==========================================
class AdaptiveDNN:
    def __init__(self, layer_dims, learning_rate=0.01):
        self.lr = learning_rate
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        
    def initialize_parameters(self):
        np.random.seed(3)
        parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            # Inicialización He/Xavier para convergencia rápida
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2./self.layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
        return parameters

    # --- FUNCIONES DE ACTIVACIÓN ---
    def relu(self, Z):
        return np.maximum(0, Z), Z
    
    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    # --- FORWARD PROPAGATION (Predicción) ---
    def forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2
        
        # Capas Ocultas (ReLU)
        for l in range(1, L):
            A_prev = A 
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A, activation_cache = self.relu(Z)
            caches.append(((A_prev, W, b), activation_cache))
            
        # Capa de Salida (Lineal para regresión, o Sigmoid para clasif.)
        # Aquí usamos Lineal pura porque predecimos Mbps (un número continuo)
        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        AL = np.dot(W, A) + b # Sin activación no-lineal al final para regresión
        caches.append(((A, W, b), None)) # Cache simple
        
        return AL, caches

    # --- BACKWARD PROPAGATION (Aprendizaje) ---
    def train_step(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        # 1. Derivada del Costo (MSE)
        dAL = 2 * (AL - Y) 
        
        # 2. Backprop Capa Salida (Lineal)
        current_cache = caches[L-1]
        linear_cache, _ = current_cache
        A_prev, W, b = linear_cache
        
        dZ = dAL # Para salida lineal, dZ = dAL (simplificado)
        grads["dW" + str(L)] = np.dot(dZ, A_prev.T) / m
        grads["db" + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        
        # 3. Backprop Capas Ocultas (ReLU)
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            linear_cache, activation_cache = current_cache
            A_prev, W, b = linear_cache
            
            dZ = self.relu_backward(dA_prev, activation_cache)
            grads["dW" + str(l + 1)] = np.dot(dZ, A_prev.T) / m
            grads["db" + str(l + 1)] = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(W.T, dZ)
            
        # 4. Update Parameters
        for l in range(len(self.parameters) // 2):
            self.parameters["W" + str(l+1)] -= self.lr * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= self.lr * grads["db" + str(l+1)]

# ==========================================
# MÓDULO 3: ORQUESTADOR
# ==========================================
def run_project():
    # 1. Datos
    env = NetworkEnvironment()
    X, Y = env.generate_traffic() # Shape (1, 2000)
    
    # 2. Arquitectura de la Red (MATRICES REALES)
    # Entrada 1 dato -> 10 neuronas -> 5 neuronas -> Salida 1 dato
    layers = [1, 10, 5, 1] 
    
    static_model = AdaptiveDNN(layers, learning_rate=0.01)
    online_model = AdaptiveDNN(layers, learning_rate=0.05) # Mayor LR para adaptarse rápido
    
    # Clonar inicialización para que sean idénticos al inicio
    import copy
    online_model.parameters = copy.deepcopy(static_model.parameters)

    # 3. Pre-entrenamiento
    print("Pre-entrenando modelos...")
    # Entrenamos con los primeros 800 datos en batch para estabilidad inicial
    X_train = X[:, :800]
    Y_train = Y[:, :800]
    
    # Epochs rápidos de calentamiento
    for i in range(50): 
        pred, caches = static_model.forward(X_train)
        static_model.train_step(pred, Y_train, caches)
        
        pred_o, caches_o = online_model.forward(X_train)
        online_model.train_step(pred_o, Y_train, caches_o)

    # 4. Loop en Tiempo Real
    results_static = []
    results_online = []
    
    print("Iniciando simulación Real-Time...")
    for t in range(X.shape[1]):
        # Dato individual (Shape 1,1) -> ¡Aquí es donde importan las matrices!
        x_t = X[:, t].reshape(1,1)
        y_t = Y[:, t].reshape(1,1)
        
        # Modelo Estático (Solo predice)
        pred_s, _ = static_model.forward(x_t)
        results_static.append(pred_s[0,0])
        
        # Modelo Online (Predice Y APRENDE)
        pred_o, caches_o = online_model.forward(x_t)
        results_online.append(pred_o[0,0])
        
        # Backprop con UN solo dato (Stochastic Gradient Descent puro)
        online_model.train_step(pred_o, y_t, caches_o)

    # 5. Gráfica
    plt.figure(figsize=(12,6))
    plt.plot(Y[0,:], color='black', alpha=0.3, label='Realidad')
    plt.plot(results_static, color='red', linestyle='--', label='Modelo Estático')
    plt.plot(results_online, color='green', label='Adaptive DNN (Online)')
    plt.axvline(x=1000, color='orange', linestyle=':')
    plt.title("Deep Neural Network Real: Static vs Online")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_project()
