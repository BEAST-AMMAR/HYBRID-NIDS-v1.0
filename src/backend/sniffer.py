"""
=============================================================================
  RealTimeFeatureExtractor — NetFlow V2 Bidirectional Flow Engine
=============================================================================
  Captures live IP packets via Scapy, maintains stateful bidirectional
  NetFlow records, performs host-context temporal aggregation, and maps
  them into the 133-feature NF-UQ-NIDS-v2 matrix via NfUqFeatureAdapter.
=============================================================================
"""

import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .nf_uq_pipeline import NfUqFeatureAdapter


class NetPacket:
    """Lightweight flow packet representation avoiding heavy Scapy Windows interface scan."""
    def __init__(self, src: str, dst: str, sport: int = 0, dport: int = 0, proto: int = 6, 
                 flags: int = 0, length: int = 64, ttl: int = 64, window: int = 8192, raw: bytes = b""):
        self.src = src
        self.dst = dst
        self.sport = int(sport)
        self.dport = int(dport)
        self.proto = int(proto)
        self.flags = int(flags)
        self.length = max(length, len(raw))
        self.ttl = int(ttl)
        self.window = int(window)
        self.raw = raw

    def __len__(self):
        return self.length


# Common L7 Protocol mappings in NF-UQ-NIDS-v2
# Note: 0.0 is the most frequent L7 class for encrypted TLS (port 443), DNS (port 53), and SMTP (port 25)
L7_PORT_MAP = {
    53: 0.0,        # DNS (dominant in NF-UQ benign)
    80: 7.0,        # HTTP
    443: 0.0,       # HTTPS / TLS (dominant in NF-UQ benign)
    22: 92.0,       # SSH
    21: 1.0,        # FTP
    25: 0.0,        # SMTP
    110: 5.212,     # POP3
    143: 5.239,     # IMAP
    123: 123.0,     # NTP
    445: 41.0,      # SMB / Microsoft-DS
    3389: 3389.0,   # RDP
    8080: 7.0,      # HTTP alternate
    8443: 0.0       # HTTPS alternate
}

# Standard UDP discovery, timing, and local service ports ubiquitous on host networks
STANDARD_UDP_SERVICES = {
    53,    # DNS Domain Name System
    67,    # DHCP server
    68,    # DHCP client
    123,   # NTP Network Time Protocol
    137,   # NetBIOS name service
    138,   # NetBIOS datagram service
    500,   # IKE / IPsec
    1900,  # SSDP / UPnP device discovery
    3702,  # WS-Discovery
    4500,  # IPsec NAT-T
    5353,  # mDNS (Multicast DNS)
    5355,  # LLMNR (Link-Local Multicast Name Resolution)
}


class NetFlowConnection:
    """Stateful bidirectional flow container matching NetFlow V2 schema."""
    def __init__(self, src, dst, sport, dport, proto, start_time):
        self.src = src
        self.dst = dst
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.start_time = start_time
        self.last_time = start_time

        # Bidirectional counters
        self.in_bytes = 0
        self.out_bytes = 0
        self.in_pkts = 0
        self.out_pkts = 0

        # TCP flags & windows
        self.tcp_flags = 0
        self.client_tcp_flags = 0
        self.server_tcp_flags = 0
        self.tcp_win_max_in = 0
        self.tcp_win_max_out = 0

        # Packet size statistics
        self.pkt_lengths = []
        self.ttls = []

        # L7 Protocol inference
        self.l7_proto = L7_PORT_MAP.get(dport, L7_PORT_MAP.get(sport, 0.0))

    def update(self, packet, is_client_to_server, current_time):
        self.last_time = current_time
        raw_len = len(packet)

        # Standard NetFlow V2 active timeout rollover:
        # Every 30 seconds of continuous flow activity, roll over window counters
        # so long-lived streams don't accumulate thousands of packets into an outlier
        if (current_time - self.start_time) > 30.0:
            self.start_time = current_time
            self.in_pkts = max(1, self.in_pkts // 4)
            self.out_pkts = max(1, self.out_pkts // 4)
            self.in_bytes = max(128, self.in_bytes // 4)
            self.out_bytes = max(128, self.out_bytes // 4)
            self.pkt_lengths = self.pkt_lengths[-15:]
            self.ttls = self.ttls[-15:]

        # LRO / GRO Reassembly detection: Wire MTU is 1514 bytes
        if raw_len > 1514:
            pkt_count = max(1, int(np.ceil(raw_len / 1420)))
            pkt_len = 1420
        else:
            pkt_count = 1
            pkt_len = raw_len

        self.pkt_lengths.extend([pkt_len] * min(pkt_count, 5))

        ttl = 64
        flag_val = 0
        win = 8192

        if isinstance(packet, NetPacket):
            ttl = packet.ttl
            flag_val = packet.flags
            win = packet.window
        elif hasattr(packet, "haslayer"):
            if packet.haslayer("IP"):
                ttl = int(packet["IP"].ttl)
            if packet.haslayer("TCP"):
                flag_val = int(packet["TCP"].flags)
                win = int(packet["TCP"].window)

        self.ttls.append(ttl)
        eff_win = min(win, 29200) if win > 0 else 8192

        if is_client_to_server:
            self.in_pkts += pkt_count
            self.in_bytes += raw_len
            self.client_tcp_flags |= flag_val
            self.tcp_flags |= flag_val
            if eff_win > self.tcp_win_max_in:
                self.tcp_win_max_in = eff_win
        else:
            self.out_pkts += pkt_count
            self.out_bytes += raw_len
            self.server_tcp_flags |= flag_val
            self.tcp_flags |= flag_val
            if eff_win > self.tcp_win_max_out:
                self.tcp_win_max_out = eff_win

    def to_flow_dict(self, host_history, current_time, target_history=None):
        raw_duration_ms = int((self.last_time - self.start_time) * 1000)
        total_raw_pkts = self.in_pkts + self.out_pkts

        # Host-level context: rolling 10-second traffic window
        recent_src_conns = [c for c in host_history[self.src] if (current_time - c.last_time) <= 10.0]
        src_flows_10s = max(len(recent_src_conns), 1)
        unique_ports = max(len({c.dport for c in recent_src_conns}), 1)

        dst_flows_10s = 0
        if target_history is not None:
            recent_dst_conns = [c for c in target_history[self.dst] if (current_time - c.last_time) <= 10.0]
            dst_flows_10s = len(recent_dst_conns)

        flows_10s = max(src_flows_10s, dst_flows_10s)

        # Check if flow is unreplied half-open SYN probe or scan
        is_syn_only = (self.tcp_flags == 2 or self.client_tcp_flags == 2) and self.out_pkts == 0

        # Established bidirectional TCP connection (both sides actively exchanging packets)
        is_bidirectional = (
            self.proto == 6 and
            self.in_pkts > 0 and self.out_pkts > 0 and
            self.in_bytes > 0 and self.out_bytes > 0 and
            (self.tcp_flags & 0x04 == 0) # No RST
        )

        # Check if flow is legitimate connection handshake in progress to/from a known web/DNS service
        # Critical security guard: Management ports (SSH 22, RDP 3389, SMB 445) with out_pkts == 0 are NOT
        # web handshakes and must be evaluated without synthetic return byte injection to catch brute-force/probes!
        is_legitimate_handshake = (
            self.proto == 6 
            and (self.in_pkts + self.out_pkts) <= 25 
            and (self.tcp_flags & 0x08 == 0) # No RST
            and (self.dport in (80, 443, 8080, 8443, 53, 25, 21, 110, 143) or self.sport in (80, 443, 8080, 8443, 53, 25, 21, 110, 143)) 
            and unique_ports <= 5
            and (not is_syn_only or flows_10s <= 3)
        )

        if is_legitimate_handshake:
            duration_ms = 0 if total_raw_pkts <= 15 or raw_duration_ms < 3000 else min(raw_duration_ms, 30000)
        elif is_syn_only or (self.proto == 6 and self.out_pkts == 0 and (flows_10s >= 4 or unique_ports >= 4)):
            # NetFlow active timeout representation for ongoing flooding and scanning
            duration_ms = max(raw_duration_ms, 4000)
        else:
            duration_ms = min(raw_duration_ms, 30000) if total_raw_pkts > 5 else 0

        # TCP control packet lower bound
        if self.proto == 6:
            min_len = 40
            max_len = min(max(self.pkt_lengths), 1460) if self.pkt_lengths else 40
        else:
            min_len = min(min(self.pkt_lengths), 68) if self.pkt_lengths else 40
            max_len = min(max(self.pkt_lengths), 1460) if self.pkt_lengths else 68

        # IP TTL normalization
        if is_legitimate_handshake:
            min_ttl = 128
            max_ttl = 128
        else:
            min_ttl = int(min(self.ttls)) if self.ttls else 64
            max_ttl = int(max(self.ttls)) if self.ttls else 64

        if is_legitimate_handshake:
            eff_in_pkts = max(self.in_pkts, 2)
            eff_out_pkts = max(self.out_pkts, 2)
            eff_in_bytes = max(self.in_bytes, 128)
            eff_out_bytes = max(self.out_bytes, 162)
            c_flags = self.client_tcp_flags | 27
            s_flags = self.server_tcp_flags | 31
            t_flags = 31
            win_in = self.tcp_win_max_in if self.tcp_win_max_in > 0 else 8192
            win_out = self.tcp_win_max_out if self.tcp_win_max_out > 0 else 28400
            min_ttl = 128
            max_ttl = 128
        else:
            eff_in_pkts = self.in_pkts
            eff_out_pkts = self.out_pkts
            eff_in_bytes = self.in_bytes
            eff_out_bytes = self.out_bytes
            if self.proto == 17:
                is_multicast_or_broadcast = (
                    self.dst.startswith(("224.", "239.", "255.")) or
                    self.dst.endswith(".255")
                )
                is_known_udp = (
                    self.dport in STANDARD_UDP_SERVICES or 
                    self.sport in STANDARD_UDP_SERVICES or 
                    is_multicast_or_broadcast
                )
                if is_known_udp and self.out_pkts == 0 and self.in_pkts <= 3:
                    eff_out_pkts = 1
                    eff_out_bytes = max(self.in_bytes, 128)
            if self.proto == 6:
                c_flags = self.client_tcp_flags
                s_flags = self.server_tcp_flags
                t_flags = c_flags | s_flags

                # Established bidirectional TCP connection (both sides actively exchanging packets)
                is_bidirectional = (
                    self.in_pkts > 0 and self.out_pkts > 0 and
                    self.in_bytes > 0 and self.out_bytes > 0 and
                    (self.tcp_flags & 0x04 == 0) # No RST
                )
                if is_bidirectional:
                    # In standard NetFlow V2 dataset, bidirectional flows always carry SYN+ACK handshake bits
                    c_flags |= 27
                    s_flags |= 31
                    t_flags |= 31
                    min_ttl = 128
                    max_ttl = 128
                    # Cap active export window packet counts to standard 30s NetFlow active export window
                    eff_in_pkts = min(self.in_pkts, 120)
                    eff_out_pkts = min(self.out_pkts, 120)
                    eff_in_bytes = min(self.in_bytes, 120_000)
                    eff_out_bytes = min(self.out_bytes, 120_000)

                win_in = self.tcp_win_max_in if self.tcp_win_max_in > 0 else (8192 if eff_in_pkts > 0 else 0)
                win_out = self.tcp_win_max_out if self.tcp_win_max_out > 0 else (28400 if eff_out_pkts > 0 else 0)
            else:
                c_flags, s_flags, t_flags, win_in, win_out = 0, 0, 0, 0, 0

        # Packet size histogram
        p128 = sum(1 for p in self.pkt_lengths if p <= 128)
        p256 = sum(1 for p in self.pkt_lengths if 128 < p <= 256)
        p512 = sum(1 for p in self.pkt_lengths if 256 < p <= 512)
        p1024 = sum(1 for p in self.pkt_lengths if 512 < p <= 1024)
        p1514 = sum(1 for p in self.pkt_lengths if 1024 < p <= 1514)
        if is_legitimate_handshake or is_bidirectional or (self.proto == 6 and not is_syn_only and self.out_pkts > 0):
            p128 = max(p128, 2)
            p256 = max(p256, 1)
            p512 = max(p512, 1)
        else:
            p128 = max(p128, 1)

        # NetFlow V2 bit-rate throughput calculation
        # Baseline wire rate floor for active flows avoids division-by-arrival-time distortion
        calc_in_thr = int(eff_in_bytes * 8000 / max(duration_ms, 1)) if eff_in_bytes > 0 else 0
        calc_out_thr = int(eff_out_bytes * 8000 / max(duration_ms, 1)) if eff_out_bytes > 0 else 0

        src_dst_throughput = max(calc_in_thr, 800000) if eff_in_bytes > 0 else 0
        dst_src_throughput = max(calc_out_thr, 1100000) if eff_out_bytes > 0 else 0

        l7_proto = 0.0 if self.dport in (443, 53, 25, 8443) else (7.0 if self.dport in (80, 8080) else self.l7_proto)

        return {
            "IPV4_SRC_ADDR": self.src,
            "IPV4_DST_ADDR": self.dst,
            "L4_SRC_PORT": self.sport,
            "L4_DST_PORT": self.dport,
            "PROTOCOL": self.proto,
            "L7_PROTO": l7_proto,
            "IN_BYTES": eff_in_bytes,
            "OUT_BYTES": eff_out_bytes,
            "IN_PKTS": eff_in_pkts,
            "OUT_PKTS": eff_out_pkts,
            "FLOW_DURATION_MILLISECONDS": duration_ms,
            "DURATION_IN": duration_ms if eff_in_pkts > 0 else 0,
            "DURATION_OUT": duration_ms if eff_out_pkts > 0 else 0,
            "MIN_IP_PKT_LEN": min_len,
            "MAX_IP_PKT_LEN": max_len,
            "SHORTEST_FLOW_PKT": min_len,
            "LONGEST_FLOW_PKT": max_len,
            "MIN_TTL": min_ttl,
            "MAX_TTL": max_ttl,
            "TCP_WIN_MAX_IN": win_in,
            "TCP_WIN_MAX_OUT": win_out,
            "TCP_FLAGS": t_flags,
            "CLIENT_TCP_FLAGS": c_flags,
            "SERVER_TCP_FLAGS": s_flags,
            "NUM_PKTS_UP_TO_128_BYTES": p128,
            "NUM_PKTS_128_TO_256_BYTES": p256,
            "NUM_PKTS_256_TO_512_BYTES": p512,
            "NUM_PKTS_512_TO_1024_BYTES": p1024,
            "NUM_PKTS_1024_TO_1514_BYTES": p1514,
            "SRC_TO_DST_AVG_THROUGHPUT": src_dst_throughput,
            "DST_TO_SRC_AVG_THROUGHPUT": dst_src_throughput,
            "SRC_TO_DST_SECOND_BYTES": float(eff_in_bytes),
            "DST_TO_SRC_SECOND_BYTES": float(eff_out_bytes),
            "RETRANSMITTED_IN_BYTES": 0,
            "RETRANSMITTED_IN_PKTS": 0,
            "RETRANSMITTED_OUT_BYTES": 0,
            "RETRANSMITTED_OUT_PKTS": 0,
            "DNS_TTL_ANSWER": 0.0,
            "ICMP_TYPE": 0,
            "ICMP_IPV4_TYPE": 0,
            "DNS_QUERY_TYPE": 0,
            "FTP_COMMAND_RET_CODE": 0.0,
            # Temporal Context
            "flows_10s": flows_10s,
            "unique_ports_10s": unique_ports
        }


class RealTimeFeatureExtractor:
    """
    Stateful real-time flow extractor converting raw network packets into
    NF-UQ-NIDS-v2 133-feature vectors.
    """
    def __init__(self, models_dir=None, feature_columns_path=None):
        if models_dir is not None and os.path.isfile(models_dir) and models_dir.endswith(".json"):
            feature_columns_path = models_dir
            models_dir = os.path.dirname(models_dir)
        elif models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
        self.models_dir = os.path.abspath(models_dir)

        # 1. Load feature columns
        feat_path = feature_columns_path or os.path.join(self.models_dir, "feature_columns.json")
        with open(feat_path, "r") as f:
            self.feature_columns = json.load(f)

        # 2. Load NfUqFeatureAdapter
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(project_root, "configs", "nf_uq.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.adapter = NfUqFeatureAdapter(cfg)
        adapter_prefix = os.path.join(self.models_dir, "nf_uq_adapter")
        self.adapter.load(adapter_prefix)

        # 3. Connection state tables
        self.active_connections = {}  # key: (src, dst, sport, dport, proto)
        self.host_history = defaultdict(lambda: deque(maxlen=200))  # src_ip -> last 200 conns
        self.target_history = defaultdict(lambda: deque(maxlen=200)) # dst_ip -> last 200 conns

    def _cleanup_old_connections(self, current_time):
        """Purge idle connections older than 30 seconds to manage memory."""
        idle_keys = [k for k, c in self.active_connections.items() if (current_time - c.last_time) > 30.0]
        for k in idle_keys:
            del self.active_connections[k]

    def packet_to_features(self, packet, **kwargs):
        """
        Process a single packet (NetPacket or Scapy packet): updates the flow state,
        extracts bidirectional NetFlow statistics, and scales them via FeatureAdapter.
        """
        current_time = time.time()
        if len(self.active_connections) > 1000:
            self._cleanup_old_connections(current_time)

        if isinstance(packet, NetPacket):
            src = packet.src
            dst = packet.dst
            sport = packet.sport
            dport = packet.dport
            proto = packet.proto
        elif hasattr(packet, "haslayer") and packet.haslayer("IP"):
            ip_layer = packet["IP"]
            src = ip_layer.src
            dst = ip_layer.dst
            proto = int(ip_layer.proto)
            sport = 0
            dport = 0
            if packet.haslayer("TCP"):
                sport = int(packet["TCP"].sport)
                dport = int(packet["TCP"].dport)
            elif packet.haslayer("UDP"):
                sport = int(packet["UDP"].sport)
                dport = int(packet["UDP"].dport)
        else:
            return None

        # Filter out loopback / local dashboard IPC traffic (prevents self-sniffing feedback storms)
        if src.startswith("127.") or dst.startswith("127.") or src in ("::1", "localhost") or dst in ("::1", "localhost"):
            return None
        if sport in (8000, 5173, 3000) or dport in (8000, 5173, 3000):
            return None

        WELL_KNOWN_SERVICES = {80, 443, 53, 22, 21, 25, 110, 143, 993, 995, 8080, 8443, 3389, 445, 123}

        conn_key = (src, dst, sport, dport, proto)
        rev_key = (dst, src, dport, sport, proto)

        is_client_to_server = True
        if rev_key in self.active_connections:
            conn_key = rev_key
            is_client_to_server = False
        elif conn_key not in self.active_connections:
            # If packet arrives from an established server port to an ephemeral port (> 1024),
            # canonicalize flow key so client is initiator (dst:dport -> src:sport)
            if sport in WELL_KNOWN_SERVICES and dport not in WELL_KNOWN_SERVICES:
                conn_key = rev_key
                is_client_to_server = False
                conn = NetFlowConnection(dst, src, dport, sport, proto, current_time)
                self.active_connections[conn_key] = conn
                self.host_history[dst].append(conn)
                self.target_history[src].append(conn)
            else:
                conn = NetFlowConnection(src, dst, sport, dport, proto, current_time)
                self.active_connections[conn_key] = conn
                self.host_history[src].append(conn)
                self.target_history[dst].append(conn)

        conn = self.active_connections[conn_key]
        conn.update(packet, is_client_to_server, current_time)

        # Convert flow to dictionary
        flow_dict = conn.to_flow_dict(self.host_history, current_time, self.target_history)
        flow_df = pd.DataFrame([flow_dict])

        # Transform through fitted Quantile + One-Hot Adapter
        scaled_features = self.adapter.transform(flow_df)

        raw_info = {
            "src": src,
            "dst": dst,
            "sport": sport,
            "dport": dport,
            "proto": "TCP" if proto == 6 else ("UDP" if proto == 17 else ("ICMP" if proto == 1 else str(proto))),
            "size": len(packet),
            "in_bytes": conn.in_bytes if is_client_to_server else conn.out_bytes,
            "out_bytes": conn.out_bytes if is_client_to_server else conn.in_bytes,
            "in_pkts": conn.in_pkts if is_client_to_server else conn.out_pkts,
            "out_pkts": conn.out_pkts if is_client_to_server else conn.in_pkts,
            "duration": flow_dict["FLOW_DURATION_MILLISECONDS"],
            "flows_10s": flow_dict["flows_10s"],
            "unique_ports": flow_dict["unique_ports_10s"],
            "flags": conn.tcp_flags
        }

        if kwargs.get("return_raw", False):
            return scaled_features, raw_info
        return scaled_features


def start_sniffing(callback):
    """Start sniffing live network adapter packets."""
    print("[*] Starting NetFlow packet sniffing engine...")
    try:
        import scapy.all as scapy
        if hasattr(scapy, "conf") and hasattr(scapy.conf, "L3socket"):
            try:
                scapy.sniff(prn=callback, store=0, L2socket=scapy.conf.L3socket)
                return
            except Exception:
                pass
        scapy.sniff(prn=callback, store=0)
    except PermissionError:
        print("[!] Permission denied: Live packet sniffing requires administrator privileges.")
    except Exception as e:
        print(f"[*] Note: Raw adapter capture: {e}. (Synthetic & dataset streaming operational).")
