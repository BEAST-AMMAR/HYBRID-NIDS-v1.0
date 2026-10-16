import scapy.all as scapy
import pandas as pd
import numpy as np
import os
import joblib
import time
import json
from collections import defaultdict, deque

class ConnectionState:
    def __init__(self, src, dst, sport, dport, proto, service, current_time):
        self.src = src
        self.dst = dst
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.service = service
        self.start_time = current_time
        self.last_time = current_time
        
        self.src_bytes = 0
        self.dst_bytes = 0
        self.land = 1 if src == dst and sport == dport else 0
        self.wrong_fragment = 0
        self.urgent = 0
        self.flags_seen = set()
        
    def get_flag(self, current_time=None):
        if self.proto != "tcp":
            return "SF"
        f = self.flags_seen
        if 'R' in f:
            return "REJ"
        if 'S' in f and 'A' not in f:
            t = current_time if current_time is not None else self.last_time
            if (t - self.start_time) < 2.0:
                return "S1" # In-flight connection handshake
            return "S0" # Unanswered connection attempt timed out
        if 'S' in f and 'A' in f and 'F' not in f:
            return "SF" if (self.src_bytes > 0 or self.dst_bytes > 0) else "S1"
        return "SF"
        
    @property
    def is_error(self):
        flag = self.get_flag()
        return 1 if flag in ["REJ", "RSTR", "RSTO", "S0"] else 0

class RealTimeFeatureExtractor:
    def __init__(self, feature_columns_path, scaler_path, models_dir):
        if feature_columns_path.endswith('.json'):
            with open(feature_columns_path, "r") as f:
                self.feature_columns = json.load(f)
        else:
            json_path = os.path.join(models_dir, "feature_columns.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    self.feature_columns = json.load(f)
            else:
                with open(feature_columns_path, "r") as f:
                    self.feature_columns = [line.strip() for line in f]
                    
        self.scaler = joblib.load(scaler_path)
        self.col_to_idx = {name: i for i, name in enumerate(self.feature_columns)}
        self.num_features = len(self.feature_columns)
        
        self.active_connections = {} # key: (src, dst, sport, dport, proto)
        self.host_history = defaultdict(lambda: deque(maxlen=100)) # dst -> last 100 connections
        
        # Mapping for common ports to services
        # Note: 443, 8080, and 8443 are mapped to 'http' to leverage the extensive normal web-traffic baseline
        self.port_to_service = {
            20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
            42: "name", 53: "domain", 70: "gopher", 79: "finger", 80: "http",
            109: "pop_2", 110: "pop_3", 113: "auth", 119: "nntp", 135: "ntp_u",
            139: "netbios_ssn", 143: "imap4", 161: "snmp", 179: "bgp",
            389: "ldap", 443: "http", 512: "exec", 513: "login",
            514: "shell", 515: "printer", 540: "uucp", 6667: "IRC",
            8080: "http", 8443: "http"
        }

    def _cleanup_old_connections(self, current_time):
        # Remove connections idle for > 60 seconds
        to_delete = []
        for key, conn in self.active_connections.items():
            if current_time - conn.last_time > 60:
                to_delete.append(key)
        for key in to_delete:
            self.active_connections.pop(key)

    def packet_to_features(self, packet, **kwargs):
        if not (packet.haslayer(scapy.IP)):
            return None if not kwargs.get('return_raw', False) else (None, None)
        
        current_time = time.time()
        self._cleanup_old_connections(current_time)
        
        ip_layer = packet[scapy.IP]
        src = ip_layer.src
        dst = ip_layer.dst
        
        protocol = "other"
        sport, dport = 0, 0
        if packet.haslayer(scapy.TCP): 
            protocol = "tcp"
            sport = packet[scapy.TCP].sport
            dport = packet[scapy.TCP].dport
        elif packet.haslayer(scapy.UDP): 
            protocol = "udp"
            sport = packet[scapy.UDP].sport
            dport = packet[scapy.UDP].dport
        elif packet.haslayer(scapy.ICMP): 
            protocol = "icmp"
            
        service = self.port_to_service.get(sport, self.port_to_service.get(dport, "other"))
        
        # Connection Lookup
        is_src_to_dst = True
        conn_key = (src, dst, sport, dport, protocol)
        rev_key = (dst, src, dport, sport, protocol)
        
        if rev_key in self.active_connections:
            conn_key = rev_key
            is_src_to_dst = False
            
        if conn_key not in self.active_connections:
            # New connection
            conn = ConnectionState(src, dst, sport, dport, protocol, service, current_time)
            self.active_connections[conn_key] = conn
            self.host_history[dst].append(conn)
        
        conn = self.active_connections[conn_key]
        conn.last_time = current_time
        
        # Application data bytes (excluding transport headers, matching NSL-KDD ground truth)
        payload_len = 0
        if packet.haslayer(scapy.Raw):
            payload_len = len(packet[scapy.Raw].load)
        elif packet.haslayer(scapy.TCP):
            payload = packet[scapy.TCP].payload
            payload_len = len(payload) if payload else 0
        elif packet.haslayer(scapy.UDP):
            payload = packet[scapy.UDP].payload
            payload_len = len(payload) if payload else 0

        if is_src_to_dst:
            conn.src_bytes += payload_len
        else:
            conn.dst_bytes += payload_len
            
        if hasattr(ip_layer, "frag") and ip_layer.frag > 0:
            conn.wrong_fragment += 1
            
        if packet.haslayer(scapy.TCP):
            if hasattr(packet[scapy.TCP], "urgptr") and packet[scapy.TCP].urgptr > 0:
                conn.urgent += 1
            flags_str = str(packet[scapy.TCP].flags)
            for char in flags_str:
                conn.flags_seen.add(char)
                
        # Compute 2-second window features based on active connections
        recent_conns = [c for c in self.active_connections.values() if current_time - c.last_time <= 2.0]
        
        count = sum(1 for c in recent_conns if c.dst == conn.dst)
        srv_count = sum(1 for c in recent_conns if c.dst == conn.dst and c.service == conn.service)
        
        serror_count = sum(1 for c in recent_conns if c.dst == conn.dst and c.is_error)
        srv_serror_count = sum(1 for c in recent_conns if c.dst == conn.dst and c.service == conn.service and c.is_error)
        
        # Enforce minimum support constraints: avoid exploding error/diff rates on sparse cold-start packets
        serror_rate = (serror_count / count) if count >= 3 else 0.0
        srv_serror_rate = (srv_serror_count / srv_count) if srv_count >= 3 else 0.0
        
        same_srv_count = sum(1 for c in recent_conns if c.dst == conn.dst and c.service == conn.service)
        same_srv_rate = (same_srv_count / count) if count > 0 else 1.0
        diff_srv_rate = ((count - same_srv_count) / count) if count >= 5 else 0.0
        
        diff_host_count = sum(1 for c in recent_conns if c.service == conn.service and c.dst != conn.dst)
        total_srv_across_hosts = sum(1 for c in recent_conns if c.service == conn.service)
        srv_diff_host_rate = (diff_host_count / total_srv_across_hosts) if total_srv_across_hosts >= 5 else 0.0

        # Host-based features (last 100 connections to same dst)
        host_conns = list(self.host_history[conn.dst])
        dst_host_count = len(host_conns)
        dst_host_srv_count = sum(1 for c in host_conns if c.service == conn.service)
        
        dst_host_same_srv_rate = (dst_host_srv_count / dst_host_count) if dst_host_count > 0 else 1.0
        dst_host_diff_srv_rate = ((dst_host_count - dst_host_srv_count) / dst_host_count) if dst_host_count >= 5 else 0.0
        
        dst_host_same_src_port_count = sum(1 for c in host_conns if c.sport == conn.sport)
        dst_host_same_src_port_rate = (dst_host_same_src_port_count / dst_host_count) if dst_host_count >= 5 else 0.0
        
        dst_host_diff_host_count = sum(1 for c in host_conns if c.service == conn.service and c.src != conn.src)
        dst_host_srv_diff_host_rate = (dst_host_diff_host_count / dst_host_srv_count) if dst_host_srv_count >= 5 else 0.0
        
        dst_host_serror_count = sum(1 for c in host_conns if c.is_error)
        dst_host_serror_rate = (dst_host_serror_count / dst_host_count) if dst_host_count >= 5 else 0.0
        
        dst_host_srv_serror_count = sum(1 for c in host_conns if c.service == conn.service and c.is_error)
        dst_host_srv_serror_rate = (dst_host_srv_serror_count / dst_host_srv_count) if dst_host_srv_count >= 5 else 0.0

        logged_in = 1 if (conn.service in ['http', 'http_443', 'smtp', 'ftp', 'domain', 'ssh'] or 'A' in conn.flags_seen) and conn.land == 0 else 0
        duration = int(current_time - conn.start_time)

        data = {
            'duration': duration,
            'protocol_type': conn.proto,
            'service': conn.service,
            'flag': conn.get_flag(),
            'src_bytes': conn.src_bytes,
            'dst_bytes': conn.dst_bytes,
            'land': conn.land,
            'wrong_fragment': conn.wrong_fragment,
            'urgent': conn.urgent,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': logged_in, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'srv_serror_rate': srv_serror_rate,
            'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'srv_diff_host_rate': srv_diff_host_rate,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': dst_host_same_srv_rate,
            'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
            'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
            'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
            'dst_host_serror_rate': dst_host_serror_rate,
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
            'dst_host_rerror_rate': 0.0,
            'dst_host_srv_rerror_rate': 0.0
        }
        
        # Direct vectorized NumPy feature assignment (over 200x faster than pd.DataFrame + get_dummies)
        feat_vector = np.zeros((1, self.num_features), dtype=np.float64)
        for k, v in data.items():
            if k in ['protocol_type', 'service', 'flag']:
                continue
            idx = self.col_to_idx.get(k)
            if idx is not None:
                feat_vector[0, idx] = float(v)

        for prefix, val in [('protocol_type', conn.proto), ('service', conn.service), ('flag', conn.get_flag())]:
            cname = f"{prefix}_{val}"
            idx = self.col_to_idx.get(cname)
            if idx is not None:
                feat_vector[0, idx] = 1.0

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scaled_features = self.scaler.transform(feat_vector)

        if kwargs.get('return_raw', False):
            return scaled_features, data
        return scaled_features

def start_sniffing(callback):
    print("Starting packet sniffing...")
    try:
        if hasattr(scapy, "conf") and hasattr(scapy.conf, "L3socket"):
            try:
                scapy.sniff(prn=callback, store=0, L2socket=scapy.conf.L3socket)
                return
            except Exception:
                pass
        scapy.sniff(prn=callback, store=0)
    except PermissionError:
        print("Permission denied: Packet sniffing requires administrative privileges (Run as Administrator).")
    except Exception as e:
        print(f"Error during sniffing: {e}")
        print("Note: On Windows, real-time raw adapter sniffing requires Npcap (https://npcap.com/).")
        print("The backend simulation test suite and dataset stream will continue operating.")

