import numpy as np
import pandas as pd

def generate_5g_traffic_data(num_samples=10000):
    """
    Generates synthetic 5G network traffic data for a specific slice (e.g., eMBB).
    
    Features:
    - Hour of Day (0-23): Traffic peaks in evenings.
    - Active User Count: Number of UEs (User Equipment) connected.
    - Packet Loss Rate (Previous Interval): Indication of current stress.
    - Latency (ms): Current network delay.
    - Bandwidth Availability (MHz): Spectrum allocated.
    
    Target:
    - Throughput Demand (Mbps): The variable we want to predict.
    """
    np.random.seed(42)
    
    # 1. Feature: Hour of Day
    hours = np.random.randint(0, 24, num_samples)
    
    # 2. Feature: Active User Count (correlated with time)
    # More users in the evening (18:00 - 22:00)
    user_base = 100
    user_variation = 100 * np.sin((hours - 14) * np.pi / 12) # Peak around evening
    users = np.abs(user_base + user_variation + np.random.normal(0, 20, num_samples)).astype(int)
    
    # 3. Feature: Latency (ms) - slightly higher when user count is high
    latency = 10 + (users / 10) + np.random.normal(0, 2, num_samples)
    
    # 4. Feature: Packet Loss Rate (0.00 to 0.05)
    pkt_loss = np.random.beta(2, 50, num_samples) 
    
    # 5. Feature: Available Bandwidth (MHz) - Random allocation between 20 and 100 MHz
    bandwidth = np.random.choice([20, 40, 60, 80, 100], num_samples)

    # GENERATING TARGET: Throughput Demand (Mbps)
    # Non-linear relationship: Users^1.5 * Bandwidth factor - Latency Penalty
    throughput = (users * 2.5) + (bandwidth * 5) - (latency * 2) + (pkt_loss * -1000)
    
    # Add some random noise/anomalies
    throughput += np.random.normal(0, 50, num_samples)
    throughput = np.maximum(throughput, 0) # Throughput cannot be negative

    # Create DataFrame
    data = pd.DataFrame({
        'Hour_of_Day': hours,
        'Active_Users': users,
        'Latency_ms': latency,
        'Packet_Loss_Rate': pkt_loss,
        'Bandwidth_MHz': bandwidth,
        'Throughput_Demand_Mbps': throughput
    })
    
    return data

# Generate and view data
df = generate_5g_traffic_data()
print("Dataset Shape:", df.shape)
print(df.head())

# Save to CSV for your use
df.to_csv('5g_slice_traffic.csv', index=False)
print("Data saved to '5g_slice_traffic.csv'")
