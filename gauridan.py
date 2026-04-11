import pandas as pd
import joblib
import numpy as np
from scapy.all import sniff, IP, TCP
import time

# 1. LOAD THE BRAIN
print("🛡️ Initializing Guardian IPS...")
model = joblib.load('general_model.pkl')
scaler = joblib.load('general_scaler.pkl')

# Global storage for tracking flows
active_flows = {}

def process_packet(pkt):
    if IP in pkt and TCP in pkt:
        src_ip = pkt[IP].src
        flow_id = f"{src_ip}_{pkt[IP].dst}_{pkt[TCP].sport}_{pkt[TCP].dport}"
        
        current_time = time.time()
        
        if flow_id not in active_flows:
            # Initialize a new flow record
            active_flows[flow_id] = {
                'start_time': current_time,
                'fwd_pkts': 1,
                'bwd_pkts': 0,
                'bytes': len(pkt),
                'last_time': current_time
            }
        else:
            # Update existing flow
            flow = active_flows[flow_id]
            flow['fwd_pkts'] += 1
            flow['bytes'] += len(pkt)
            duration = current_time - flow['start_time']
            
            # Every 10 packets, let's check if it's an attack
            if flow['fwd_pkts'] % 10 == 0:
                # Prepare the Golden 5 Features
                # [Duration, Fwd Packets, Bwd Packets, Packets/s, Bytes/s]
                duration_ms = duration * 1000
                pkt_rate = flow['fwd_pkts'] / (duration if duration > 0 else 0.1)
                byte_rate = flow['bytes'] / (duration if duration > 0 else 0.1)
                
                features = np.array([[
                    duration_ms, 
                    flow['fwd_pkts'], 
                    flow['bwd_pkts'], 
                    pkt_rate, 
                    byte_rate
                ]])
                
                # Scale and Predict
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)
                
                if prediction[0] == 1:
                    print(f"🚨 ALERT: Malicious activity detected from {src_ip}!")
                    print(f"Action: LOGGING ATTACK - [Feature Stats: {features.tolist()}]")
                    # In a real production server, you'd trigger: os.system(f"iptables -A INPUT -s {src_ip} -j DROP")
                else:
                    print(f"✅ Safe: Traffic from {src_ip} verified.")

# 3. START SNIFFING
print("📡 Monitoring network traffic on interface eth0...")
# Note: Use iface="wlan0" or "eth0" depending on your OS (Check 'ifconfig')
sniff(filter="tcp", prn=process_packet, store=0)