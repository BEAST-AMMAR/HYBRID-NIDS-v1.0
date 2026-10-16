import time
import random
import threading
import scapy.all as scapy
import queue
import logging

class SyntheticTrafficGenerator:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.virtual = True
        self.packet_queue = None

    def _generate_normal_flow(self):
        """Generates realistic normal flow packets with valid handshakes and protocols"""
        src_ip = f"192.168.1.{random.randint(10, 150)}"
        dst_ip = f"192.168.1.{random.randint(151, 254)}"
        sport = random.randint(10000, 60000)
        proto_choice = random.choice(['http', 'https', 'dns', 'smtp'])
        
        pkts = []
        if proto_choice == 'dns':
            # UDP DNS Query and Reply
            query = scapy.IP(src=src_ip, dst="192.168.1.1") / scapy.UDP(sport=sport, dport=53) / scapy.Raw(load=b"\xaa\xbb\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x06google\x03com\x00\x00\x01\x00\x01")
            reply = scapy.IP(src="192.168.1.1", dst=src_ip) / scapy.UDP(sport=53, dport=sport) / scapy.Raw(load=b"\xaa\xbb\x81\x80\x00\x01\x00\x01\x00\x00\x00\x00\x06google\x03com\x00\x00\x01\x00\x01\xc0\x0c\x00\x01\x00\x01\x00\x00\x01\x2c\x00\x04\xac\xd9\x05\xbe")
            pkts.extend([query, reply])
        else:
            # TCP established flow (HTTP, HTTPS, SMTP)
            dport = 80 if proto_choice == 'http' else (443 if proto_choice == 'https' else 25)
            # Step 1: Client SYN
            syn = scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=sport, dport=dport, flags='S', seq=1000)
            # Step 2: Server SYN-ACK
            syn_ack = scapy.IP(src=dst_ip, dst=src_ip) / scapy.TCP(sport=dport, dport=sport, flags='SA', seq=2000, ack=1001)
            # Step 3: Client ACK with payload
            payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n" if dport in (80, 443) else b"EHLO client.example.com\r\n"
            ack_data = scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=sport, dport=dport, flags='PA', seq=1001, ack=2001) / scapy.Raw(load=payload)
            pkts.extend([syn, syn_ack, ack_data])
            
        return pkts

    def _generate_anomaly_packet(self):
        """Generates anomalous packets that trigger ML features like 'land' or high 'count'"""
        anomaly_type = random.choice(['syn_flood', 'land_attack', 'port_scan', 'large_payload'])
        
        if anomaly_type == 'syn_flood':
            # SYN flood: Generate a BURST of packets to trigger high `count` and `serror_rate`
            pkts = []
            dst_ip = "192.168.1.100"
            for _ in range(35):
                src_ip = f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
                pkts.append(scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=random.randint(1024, 65535), dport=80, flags='S'))
            return pkts
            
        elif anomaly_type == 'land_attack':
            # LAND attack: source IP == destination IP, which sets `land=1` in sniffer
            ip = "192.168.1.100"
            return [scapy.IP(src=ip, dst=ip) / scapy.TCP(sport=80, dport=80, flags='S')]
            
        elif anomaly_type == 'large_payload':
            # Huge packet (DoS)
            src_ip = "192.168.1.50"
            dst_ip = "192.168.1.100"
            payload = b"X" * 60000 # Very large payload
            return [scapy.IP(src=src_ip, dst=dst_ip) / scapy.UDP(sport=1234, dport=80) / scapy.Raw(load=payload)]
            
        else: # port_scan
            src_ip = "192.168.1.50" 
            dst_ip = "192.168.1.100"
            return [scapy.IP(src=src_ip, dst=dst_ip) / scapy.TCP(sport=54321, dport=p, flags='S') for p in range(20, 36)]

    def _injection_loop(self):
        print(f"Started Synthetic Generator (Virtual={self.virtual})")
        while self.is_running:
            # Dynamic mix: variable 10-25% chance of an anomaly burst, otherwise batch of normal flows
            anomaly_threshold = random.uniform(0.10, 0.25)
            is_anomaly = random.random() < anomaly_threshold
            
            if is_anomaly:
                pkts = self._generate_anomaly_packet()
            else:
                pkts = []
                for _ in range(random.randint(1, 2)):
                    pkts.extend(self._generate_normal_flow())
                
            for pkt in pkts:
                if self.virtual and self.packet_queue is not None:
                    try:
                        self.packet_queue.put_nowait(pkt)
                    except queue.Full:
                        pass
                elif not self.virtual:
                    try:
                        scapy.send(pkt, verbose=False)
                    except Exception as e:
                        print(f"Failed to inject packet physically: {e}")
                    
            # Variable sleep to simulate realistic network delay
            time.sleep(random.uniform(0.04, 0.12))

    def start(self, virtual=True, target_queue=None):
        self.virtual = virtual
        if target_queue is not None:
            self.packet_queue = target_queue
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._injection_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)

generator_instance = SyntheticTrafficGenerator()
