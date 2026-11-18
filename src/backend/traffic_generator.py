import time
import random
import threading
import queue
import logging
from .sniffer import NetPacket


class SyntheticTrafficGenerator:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.virtual = True
        self.packet_queue = None

    def _generate_normal_flow(self):
        """Generates realistic normal flow packets with valid handshakes and protocols."""
        src_ip = f"192.168.1.{random.randint(10, 150)}"
        dst_ip = f"192.168.1.{random.randint(151, 254)}"
        sport = random.randint(10000, 60000)
        proto_choice = random.choice(['http', 'https', 'https', 'dns', 'dns', 'smtp'])
        
        pkts = []
        if proto_choice == 'dns':
            # UDP DNS Query and Reply (proto=17)
            query = NetPacket(src=src_ip, dst="192.168.1.1", sport=sport, dport=53, proto=17, length=78)
            reply = NetPacket(src="192.168.1.1", dst=src_ip, sport=53, dport=sport, proto=17, length=142)
            pkts.extend([query, reply])
        else:
            # TCP established flow (proto=6) with full handshake and bidirectional payload
            dport = 80 if proto_choice == 'http' else (443 if proto_choice == 'https' else 25)
            # Step 1: Client SYN (flags=2)
            syn = NetPacket(src=src_ip, dst=dst_ip, sport=sport, dport=dport, proto=6, flags=2, length=64)
            # Step 2: Server SYN-ACK (flags=18, window=28400)
            syn_ack = NetPacket(src=dst_ip, dst=src_ip, sport=dport, dport=sport, proto=6, flags=18, length=64, window=28400)
            # Step 3: Client ACK (flags=16)
            client_ack = NetPacket(src=src_ip, dst=dst_ip, sport=sport, dport=dport, proto=6, flags=16, length=54)
            # Step 4: Client Request (flags=24)
            req_len = random.randint(350, 800) if dport in (80, 443) else 120
            client_req = NetPacket(src=src_ip, dst=dst_ip, sport=sport, dport=dport, proto=6, flags=24, length=req_len)
            
            pkts.extend([syn, syn_ack, client_ack, client_req])

            # Step 5: Server Data Chunks (flags=24)
            num_chunks = random.randint(3, 8) if dport in (80, 443) else 1
            for _ in range(num_chunks):
                data_len = random.randint(1024, 1460)
                server_chunk = NetPacket(src=dst_ip, dst=src_ip, sport=dport, dport=sport, proto=6, flags=24, length=data_len, window=28400)
                pkts.append(server_chunk)

            # Step 6: Client Final ACK
            pkts.append(NetPacket(src=src_ip, dst=dst_ip, sport=sport, dport=dport, proto=6, flags=16, length=54))
            
        return pkts

    def _generate_anomaly_packet(self):
        """Generates anomalous packets that trigger NetFlow V2 detection features."""
        anomaly_type = random.choice(['syn_flood', 'port_scan', 'brute_force', 'large_payload', 'bot_beacon'])
        
        if anomaly_type == 'syn_flood':
            # DDoS / SYN flood: burst of SYN packets from random IPs to target
            pkts = []
            dst_ip = "192.168.1.100"
            for _ in range(25):
                src_ip = f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
                pkts.append(NetPacket(src=src_ip, dst=dst_ip, sport=random.randint(1024, 65535), dport=80, proto=6, flags=2, length=64))
            return pkts
            
        elif anomaly_type == 'port_scan':
            # Port scanning across multiple ports from single source IP
            src_ip = f"192.168.1.{random.randint(200, 250)}"
            dst_ip = "192.168.1.100"
            ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 1433, 3306, 3389, 8080]
            return [NetPacket(src=src_ip, dst=dst_ip, sport=54321, dport=p, proto=6, flags=2, length=64) for p in ports]
            
        elif anomaly_type == 'brute_force':
            # High frequency authentication flows on SSH / RDP
            pkts = []
            src_ip = f"192.168.1.{random.randint(200, 250)}"
            dst_ip = "192.168.1.100"
            sport = random.randint(10000, 60000)
            dport = random.choice([22, 3389])
            for seq in range(10):
                syn = NetPacket(src=src_ip, dst=dst_ip, sport=sport + seq, dport=dport, proto=6, flags=2, length=64)
                pkts.append(syn)
            return pkts

        elif anomaly_type == 'bot_beacon':
            # Subtle keepalive beacon to external C2 server
            src_ip = f"192.168.1.{random.randint(10, 150)}"
            c2_ip = f"{random.randint(45, 190)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            c2_port = random.choice([6667, 8888, 6666])
            pkt = NetPacket(src=src_ip, dst=c2_ip, sport=random.randint(20000, 60000), dport=c2_port, proto=6, flags=24, length=96)
            return [pkt]

        else:  # large_payload / volumetric DoS
            src_ip = "192.168.1.50"
            dst_ip = "192.168.1.100"
            pkts = [NetPacket(src=src_ip, dst=dst_ip, sport=1234, dport=80, proto=17, length=1420) for _ in range(15)]
            return pkts

    def _injection_loop(self):
        print(f"[*] Started Synthetic NetFlow Generator (Virtual={self.virtual})")
        while self.is_running:
            # Realistic network proportion: 4-7% chance of an attack burst, otherwise batches of sustained normal traffic
            is_anomaly = (random.random() < 0.05)
            
            if is_anomaly:
                pkts = self._generate_anomaly_packet()
            else:
                pkts = []
                for _ in range(random.randint(2, 4)):
                    pkts.extend(self._generate_normal_flow())
                
            for pkt in pkts:
                if self.virtual and self.packet_queue is not None:
                    try:
                        self.packet_queue.put_nowait(pkt)
                    except queue.Full:
                        pass
                elif not self.virtual:
                    try:
                        import scapy.all as scapy
                        sc_pkt = scapy.IP(src=pkt.src, dst=pkt.dst)
                        if pkt.proto == 6:
                            sc_pkt = sc_pkt / scapy.TCP(sport=pkt.sport, dport=pkt.dport, flags=pkt.flags)
                        elif pkt.proto == 17:
                            sc_pkt = sc_pkt / scapy.UDP(sport=pkt.sport, dport=pkt.dport)
                        scapy.send(sc_pkt, verbose=False)
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
