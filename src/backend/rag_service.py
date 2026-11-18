import os
import json
import re
import hashlib
from typing import List, Dict, Any, Optional

from .rag_knowledge import SimpleRAG


class RAGExplanationService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-placeholder")
        
        # Load comprehensive knowledge base dynamically
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.raw_entries: List[Dict[str, Any]] = []
        self._load_knowledge_base()

        # Build comprehensive feature map covering all 133 NetFlow V2 features
        self.feature_map = self._build_feature_map()

        # Initialize lightweight dependency-free RAG index
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        kb_file = os.path.join(project_root, "knowledge_base.json")
        self.simple_rag = SimpleRAG(kb_file) if os.path.exists(kb_file) else None

        # Optional vector index if available
        self.kb_index = None
        try:
            from .kb_index import KBIndex
            if os.path.exists(kb_file):
                self.kb_index = KBIndex(kb_file)
                try:
                    self.kb_index.load()
                except Exception:
                    pass
        except Exception:
            self.kb_index = None

    def _load_knowledge_base(self):
        """Loads and indexes knowledge base entries from JSON into dynamic lookup dictionaries."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        kb_path = os.path.join(project_root, "knowledge_base.json")

        self.raw_entries = []
        if os.path.exists(kb_path):
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    self.raw_entries = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load knowledge base JSON: {e}")

        for entry in self.raw_entries:
            name = entry.get("name", "")
            attack_type = str(entry.get("attack_type", "")).strip()
            category = str(entry.get("category", attack_type)).strip().lower()
            slug = re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))

            item = {
                "name": name,
                "attack_type": attack_type,
                "category": category,
                "description": entry.get("description", f"Network threat in category {attack_type}."),
                "indicators": entry.get("indicators", ["Anomaly score threshold exceeded"]),
                "mitigation": entry.get("mitigation", "Inspect host and filter offending source IP."),
            }

            # Register multiple lookup aliases for instant matching
            self.knowledge_base[slug] = item
            self.knowledge_base[attack_type.lower()] = item
            self.knowledge_base[category] = item
            self.knowledge_base[attack_type] = item

    def _build_feature_map(self) -> Dict[str, str]:
        """Provides human-readable cybersecurity explanations for NetFlow V2 features."""
        return {
            # Core Flow Volume & Duration
            "FLOW_DURATION_MILLISECONDS": "Total duration of the network flow in milliseconds",
            "DURATION_IN": "Duration of incoming client-to-server traffic in milliseconds",
            "DURATION_OUT": "Duration of outgoing server-to-client response in milliseconds",
            "IN_BYTES": "Total volume of incoming bytes sent from source to destination",
            "OUT_BYTES": "Total volume of response bytes sent from destination to source",
            "IN_PKTS": "Total number of packets sent from source to destination",
            "OUT_PKTS": "Total number of packets sent from destination to source",
            "SRC_TO_DST_AVG_THROUGHPUT": "Average network throughput rate from source to destination (bps)",
            "DST_TO_SRC_AVG_THROUGHPUT": "Average response throughput rate from destination to source (bps)",
            "SRC_TO_DST_SECOND_BYTES": "Data transfer rate observed per second",
            "DST_TO_SRC_SECOND_BYTES": "Response transfer rate observed per second",

            # Packet Length Characteristics
            "MIN_IP_PKT_LEN": "Minimum IP packet length observed in this flow",
            "MAX_IP_PKT_LEN": "Maximum IP packet length observed in this flow",
            "SHORTEST_FLOW_PKT": "Shortest payload size transferred",
            "LONGEST_FLOW_PKT": "Longest payload size transferred (key indicator for buffer exploits)",
            "NUM_PKTS_UP_TO_128_BYTES": "Count of small packets (<=128 bytes, characteristic of scans & SYN floods)",
            "NUM_PKTS_128_TO_256_BYTES": "Count of small-to-medium packets (128-256 bytes)",
            "NUM_PKTS_256_TO_512_BYTES": "Count of medium packets (256-512 bytes)",
            "NUM_PKTS_512_TO_1024_BYTES": "Count of large packets (512-1024 bytes)",
            "NUM_PKTS_1024_TO_1514_BYTES": "Count of full MTU packets (1024-1514 bytes, standard for file transfers/floods)",

            # TCP Flags & States
            "TCP_FLAGS": "Cumulative bitwise TCP flags observed across the entire flow",
            "CLIENT_TCP_FLAGS": "TCP flags initiated by client (SYN, ACK, PSH, FIN, RST)",
            "SERVER_TCP_FLAGS": "TCP flags returned by responder",
            "TCP_FLAGS_SYN": "TCP SYN connection synchronization flag set",
            "TCP_FLAGS_FIN": "TCP FIN connection termination flag set",
            "TCP_FLAGS_RST": "TCP RST connection reset flag set (indicates aborted/closed port)",
            "TCP_FLAGS_PSH": "TCP PUSH flag set (immediate data push to application layer)",
            "CLIENT_TCP_FLAGS_SYN": "Client-initiated SYN connection handshake flag",
            "SERVER_TCP_FLAGS_SYN": "Server SYN-ACK handshake response flag",
            "TCP_WIN_MAX_IN": "Maximum client TCP window size advertised",
            "TCP_WIN_MAX_OUT": "Maximum server TCP window size advertised",
            "MIN_TTL": "Minimum Time-To-Live hop count (helps identify spoofed or proxied paths)",
            "MAX_TTL": "Maximum Time-To-Live hop count",
            "RETRANSMITTED_IN_BYTES": "Volume of retransmitted incoming bytes (indicates network congestion or MITM)",

            # Protocol & Port Descriptors
            "proto_TCP": "Transmission Control Protocol (TCP, Protocol 6)",
            "proto_UDP": "User Datagram Protocol (UDP, Protocol 17)",
            "proto_ICMP": "Internet Control Message Protocol (ICMP, Protocol 1)",
            "proto_other": "Encapsulated or non-standard transport protocol",
            "port_53": "DNS Domain Name Service resolution port",
            "port_80": "HTTP unencrypted Web server port",
            "port_443": "HTTPS TLS/SSL encrypted Web communication port",
            "port_22": "SSH Secure Shell remote administrative port",
            "port_21": "FTP File Transfer Protocol control port",
            "port_445": "SMB Microsoft Active Directory / file sharing port",
            "port_3389": "RDP Microsoft Remote Desktop Protocol port",
            "port_8080": "Alternative HTTP Proxy / Web application port",
            "port_other": "Non-standard or dynamic high-ephemeral port",

            # Temporal / Host Context
            "flows_10s": "Number of flows generated by this source host in the past 10 seconds",
            "unique_ports_10s": "Number of unique destination ports targeted in the past 10 seconds"
        }

    def _is_telemetry_consistent(self, category: str, info: dict) -> bool:
        """
        Validates that high-severity attack classifications are physically supported
        by actual NetFlow telemetry data before allowing them to be reported.
        Prevents semantic search embedding artifacts from mislabeling normal small flows.
        """
        cat = category.lower()
        in_pkts = int(info.get("in_pkts", 1))
        out_pkts = int(info.get("out_pkts", 0))
        in_bytes = int(info.get("in_bytes", info.get("size", 0)))
        out_bytes = int(info.get("out_bytes", 0))
        flows_10s = int(info.get("flows_10s", 1))
        unique_ports = int(info.get("unique_ports", info.get("unique_ports_10s", 1)))
        dport = int(info.get("dport", 0))
        sport = int(info.get("sport", 0))

        if cat in ("dos", "ddos"):
            # DoS/DDoS physically requires high burst volume, packet rate, or high flows_10s with asymmetric/unreplied traffic
            has_packet_burst = in_pkts >= 20 or in_bytes >= 15_000
            has_rapid_flows = flows_10s >= 8 and (out_bytes == 0 or out_pkts == 0)
            is_unreplied = (out_pkts == 0 or out_bytes < 500 or in_bytes > 10 * max(out_bytes, 1))
            return (has_packet_burst or has_rapid_flows) and is_unreplied

        if cat in ("scanning", "reconnaissance"):
            # Scanning physically requires scanning multiple ports or high multi-flow probe rates
            return unique_ports >= 3 or flows_10s >= 6

        if cat == "brute_force":
            AUTH_PORTS = (22, 3389, 21, 23, 1433, 3306, 5432)
            is_auth_port = (dport in AUTH_PORTS or sport in AUTH_PORTS)
            return is_auth_port and (flows_10s >= 4 or in_pkts >= 10)

        if cat in ("theft", "exfiltration"):
            return out_bytes >= 100_000 or in_bytes >= 500_000

        if cat in ("ransomware", "worms"):
            return (dport in (445, 139) or sport in (445, 139)) and (in_bytes > 5_000 or flows_10s >= 3)

        if cat == "shellcode":
            SHELL_PORTS = (1347, 4444, 31337, 1234, 6666, 9001)
            return dport in SHELL_PORTS or sport in SHELL_PORTS

        return True

    def classify(self, info: dict, top_features: List[str] = None) -> tuple:
        """Public classification method returning (attack_key, display_name)."""
        attack_key = self._classify_anomaly(info, top_features)
        context = (
            self.knowledge_base.get(attack_key)
            or self.knowledge_base.get(attack_key.lower())
            or self.knowledge_base.get("general_anomaly", {})
        )
        name = context.get("name", context.get("attack_type", attack_key.replace("_", " ").title()))
        return attack_key, name

    def _classify_anomaly(self, info: dict, top_features: List[str] = None) -> str:
        """
        Classifies NetFlow anomalies into the NF-UQ-NIDS-v2 attack categories
        using a STRICT, ordered rule chain with calibrated thresholds.

        Rules are ordered from most-specific to most-general to prevent
        over-classification. Normal traffic patterns are explicitly whitelisted first.
        """
        # 0. Check explicit dataset label (highest priority — from CSV/PCAP evaluation)
        explicit_label = str(info.get("attack_label", info.get("attack_type", info.get("attack", "")))).strip()
        if explicit_label and explicit_label.lower() not in ("normal", "benign", "none", ""):
            lower_label = explicit_label.lower()
            if lower_label in self.knowledge_base:
                return lower_label
            if explicit_label in self.knowledge_base:
                return explicit_label

        # If explicitly marked as normal or benign, return normal
        if info.get("is_anomaly") is False:
            return "normal"

        # Extract flow telemetry
        features_str = " ".join(top_features).lower() if top_features else ""
        proto = str(info.get("proto", "TCP")).upper()
        dport = int(info.get("dport", 0))
        sport = int(info.get("sport", 0))
        in_bytes = int(info.get("in_bytes", info.get("size", 0)))
        out_bytes = int(info.get("out_bytes", 0))
        in_pkts = int(info.get("in_pkts", 1))
        out_pkts = int(info.get("out_pkts", 0))
        duration_ms = int(info.get("duration", info.get("duration_ms", 0)))
        flows_10s = int(info.get("flows_10s", 1))
        unique_ports = int(info.get("unique_ports", info.get("unique_ports_10s", 1)))
        flags = int(info.get("flags", 0))
        score = float(info.get("score", 0.5))

        # -----------------------------------------------------------------------
        # Rule 1: Port Scanning & Network Enumeration (HIGHEST HEURISTIC PRIORITY)
        # Evaluated before single-port rules to prevent a multi-port scan touching
        # port 22 or port 445 from being misclassified as brute force or worms!
        # Requires: multiple distinct ports targeted with low or unreplied response
        # -----------------------------------------------------------------------
        if unique_ports >= 4 or (unique_ports >= 3 and flows_10s >= 3):
            if out_bytes < 500 or out_pkts <= 1:
                return "scanning"

        # -----------------------------------------------------------------------
        # Rule 2: Network Reconnaissance (ICMP sweep or wide-subnet scan)
        # -----------------------------------------------------------------------
        if proto == "ICMP" and (flows_10s >= 8 or unique_ports >= 4):
            return "reconnaissance"
        if unique_ports >= 8 and flows_10s >= 10:
            return "reconnaissance"

        # -----------------------------------------------------------------------
        # Rule 3: Volumetric Floods (DoS / DDoS & SYN Floods)
        # -----------------------------------------------------------------------
        is_syn = (flags & 0x02) != 0 or "syn" in features_str or flags == 2
        # SYN flood: TCP SYN connection attempts with high burst rate and unreplied handshake
        if proto == "TCP" and is_syn and (flows_10s >= 8 or in_pkts >= 15) and (out_bytes == 0 or out_pkts == 0):
            return "ddos" if flows_10s >= 20 else "dos"

        # High-volume multi-source DDoS burst
        if (in_pkts > 80 or in_bytes > 80_000) and flows_10s >= 15:
            return "ddos"

        # Single-source high-rate DoS (large payload or fast packet flood with unreplied response)
        is_unreplied_flood = (out_bytes == 0 or out_pkts <= 1 or in_bytes > 15 * max(out_bytes, 1))
        if in_pkts >= 100 and is_unreplied_flood:
            return "dos"
        if in_pkts >= 20 and duration_ms < 3000 and (out_bytes == 0 or out_pkts == 0):
            return "dos"
        if in_bytes > 25_000 and duration_ms < 2000 and out_bytes < 500:
            return "dos"

        # -----------------------------------------------------------------------
        # Rule 4: Data Exfiltration / Theft (Massive unauthorized outbound egress)
        # -----------------------------------------------------------------------
        # Theft requires outbound data exfiltration: source sends massive payload (>10MB)
        # to a non-standard external port with minimal/no response bytes
        if in_bytes > 10_000_000 and out_bytes < 50_000 and dport not in (80, 443, 8080, 8443):
            return "theft"

        # -----------------------------------------------------------------------
        # Rule 5: Authentication Brute Force
        # TARGETED: requires authentication port AND focused targets (unique_ports <= 2)
        # -----------------------------------------------------------------------
        AUTH_PORTS = (22, 3389, 21, 23, 1433, 3306, 5432)
        if (dport in AUTH_PORTS or sport in AUTH_PORTS) and unique_ports <= 2:
            if flows_10s >= 4 or in_pkts >= 15:
                return "brute_force"

        # -----------------------------------------------------------------------
        # Rule 6: Ransomware / Worm (SMB port 445 / 139)
        # TARGETED: requires unique_ports <= 2
        # -----------------------------------------------------------------------
        if (dport in (445, 139) or sport in (445, 139)) and unique_ports <= 2:
            if in_bytes > 10_000 or out_bytes > 10_000:
                return "ransomware"
            elif flows_10s >= 3 or in_pkts >= 8:
                return "worms"

        # -----------------------------------------------------------------------
        # Rule 7: Shellcode staging on well-known exploit ports
        # -----------------------------------------------------------------------
        if dport in (1347, 4444, 31337, 1234, 6666, 9001) or sport in (4444, 31337):
            return "shellcode"

        # -----------------------------------------------------------------------
        # Rule 8: BGP / Routing protocol fuzzing
        # -----------------------------------------------------------------------
        if dport == 179 or sport == 179:
            return "fuzzers"

        # -----------------------------------------------------------------------
        # Rule 9: MITM / TTL anomalies
        # -----------------------------------------------------------------------
        if "retransmitted_in_bytes" in features_str and ("min_ttl" in features_str or "max_ttl" in features_str):
            return "mitm"

        # -----------------------------------------------------------------------
        # Rule 10: DNS Tunneling / Infiltration (UDP 53 with oversized payload)
        # -----------------------------------------------------------------------
        if dport == 53 and proto == "UDP":
            if in_bytes > 512 or in_pkts > 5:
                return "infilteration"
            return "generic"

        # -----------------------------------------------------------------------
        # Rule 11: Botnet C2 Beaconing
        # -----------------------------------------------------------------------
        INTERNAL_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                             "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                             "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                             "172.29.", "172.30.", "172.31.")
        dst = str(info.get("dst", ""))
        is_external_dst = not any(dst.startswith(p) for p in INTERNAL_PREFIXES) and dst not in ("unknown", "", "127.0.0.1", "localhost")

        if dport == 8080 and is_external_dst and in_bytes < 500:
            if duration_ms > 10_000 or in_pkts > 20:
                return "backdoor"
            return "bot"

        # Tiny persistent beacons to external IPs on non-standard ports or SSL
        if is_external_dst and in_bytes < 300 and out_bytes < 300 and (duration_ms > 20_000 or (dport not in (80, 53) and in_pkts <= 2)):
            return "bot"

        # -----------------------------------------------------------------------
        # Rule 12: Web Application Attacks (port 80/443/8080)
        # -----------------------------------------------------------------------
        if dport in (80, 443, 8080) or sport in (80, 443):
            # Injection: mid-sized anomalous payloads
            if in_bytes > 8_000 and in_bytes < 100_000 and "longest_flow_pkt" in features_str:
                return "injection"
            # Large payload exploits
            if in_bytes >= 100_000 and "longest_flow_pkt" in features_str:
                return "exploits"
            # Password spraying: many small requests in rapid succession
            if flows_10s >= 10 and in_bytes < 2_000 and in_pkts >= 8:
                return "password"
            # XSS patterns: repeated medium requests
            if flows_10s >= 5 and 500 < in_bytes < 3_000 and in_pkts >= 5 and out_bytes > 0:
                return "xss"

        # -----------------------------------------------------------------------
        # Rule 13: Neural Semantic & Lexical RAG Fallback Search (Guarded by Physical Telemetry)
        # -----------------------------------------------------------------------
        query_text = f"{proto} port {dport} {in_bytes} bytes {in_pkts} packets {flows_10s} flows {features_str}"

        # Step A: Neural vector cosine similarity via KBIndex
        if self.kb_index is not None:
            try:
                results = self.kb_index.query(query_text, top_k=1)
                if results and results[0][1] >= 0.50:
                    entry = results[0][0]
                    matched_type = str(entry.get("category", entry.get("attack_type", "general_anomaly"))).lower()
                    if matched_type not in ("normal", "benign", "general_anomaly"):
                        if self._is_telemetry_consistent(matched_type, info):
                            return matched_type
            except Exception:
                pass

        # Step B: Lexical TF-IDF fallback via SimpleRAG
        if self.simple_rag is not None:
            try:
                results = self.simple_rag.query(query_text, top_k=1)
                if results and results[0][1] > 0.20:
                    matched_type = str(results[0][0].get("category", results[0][0].get("attack_type", "general_anomaly"))).lower()
                    if matched_type not in ("generic", "general_anomaly", "normal"):
                        if self._is_telemetry_consistent(matched_type, info):
                            return matched_type
            except Exception:
                pass

        return "general_anomaly"

    def _generate_behavioral_narrative(self, attack_key: str, info: dict) -> str:
        """Synthesizes dynamic, context-aware cybersecurity analysis using actual telemetry facts."""
        proto = str(info.get("proto", "TCP")).upper()
        dport = int(info.get("dport", 0))
        sport = int(info.get("sport", 0))
        dst = str(info.get("dst", "unknown"))
        src = str(info.get("src", "unknown"))
        size = int(info.get("size", info.get("in_bytes", 0)))
        out_bytes = int(info.get("out_bytes", 0))
        in_pkts = int(info.get("in_pkts", 1))
        flows_10s = int(info.get("flows_10s", 1))
        duration_ms = int(info.get("duration", info.get("duration_ms", 0)))
        unique_ports = int(info.get("unique_ports", info.get("unique_ports_10s", 1)))

        if attack_key in ("scanning", "reconnaissance"):
            return (
                f"Network probe detected: `{src}` targeted {unique_ports} distinct port(s) "
                f"across {flows_10s} rapid flow(s) with {size:,} bytes sent and only {out_bytes:,} bytes returned. "
                f"The zero or minimal return-byte pattern (no bidirectional handshake completion) is characteristic "
                f"of automated scanning utilities (e.g. Nmap SYN stealth probe, Masscan) mapping active listening services."
            )
        elif attack_key == "ddos":
            return (
                f"Distributed flood targeting `{dst}:{dport}` via **{proto}**: {in_pkts:,} inbound packets, "
                f"{size:,} bytes across {flows_10s} concurrent source flows in a 10-second window. "
                f"The volumetric convergence from multiple distributed origins significantly exceeds "
                f"legitimate traffic baselines, causing socket table saturation and service degradation."
            )
        elif attack_key == "dos":
            return (
                f"Single-source flood from `{src}` targeting `{dst}:{dport}`: {in_pkts:,} packets / {size:,} bytes "
                f"within {duration_ms}ms. The extreme packet rate from a single origin with no bidirectional "
                f"application response indicates an attempt to exhaust connection-tracking resources or buffer capacity."
            )
        elif attack_key == "brute_force":
            return (
                f"Authentication attack on administrative port `{dport}` ({proto}) from `{src}`: "
                f"{flows_10s} consecutive connection attempts within 10 seconds, {in_pkts:,} total packets. "
                f"The high-frequency reconnection pattern with short-lived flows is characteristic of automated "
                f"credential dictionary attacks or SSH brute-force tooling."
            )
        elif attack_key == "theft":
            return (
                f"Data exfiltration signature from `{src}` to `{dst}:{dport}` via {proto}: "
                f"{size:,} inbound bytes with only {out_bytes:,} outbound bytes (asymmetric egress pattern). "
                f"The sustained high-volume outward transfer deviates sharply from normal interactive session baselines "
                f"and matches data staging or cloud-upload exfiltration behavior."
            )
        elif attack_key in ("injection", "xss"):
            return (
                f"Web application attack payload to `{dst}:{dport}` via {proto}: {size:,} bytes "
                f"with {in_pkts:,} request packets across {flows_10s} flows. "
                f"The payload size distribution and request rate deviate from standard browser HTTP/S traffic, "
                f"exhibiting characteristics of SQL injection, XSS injection, or automated web fuzzing patterns."
            )
        elif attack_key == "exploits":
            return (
                f"Exploit payload targeting `{dst}:{dport}` via {proto}: {size:,} bytes in {in_pkts:,} packets. "
                f"The oversized application payload with non-standard structure indicates a structured buffer overflow "
                f"or CVE exploit attempt targeting service vulnerabilities."
            )
        elif attack_key in ("bot", "backdoor"):
            return (
                f"Botnet C2 beaconing from `{src}` to `{dst}:{dport}` via {proto}: "
                f"{size:,} bytes / {in_pkts:,} packets over {duration_ms}ms. "
                f"This compact persistent transmission profile with external non-standard destination matches "
                f"command-and-control heartbeat beaconing or reverse shell maintenance traffic."
            )
        elif attack_key in ("ransomware", "worms"):
            return (
                f"Lateral malware propagation via SMB/RPC port `{dport}` from `{src}`: "
                f"{size:,} bytes, {in_pkts:,} packets in {flows_10s} flows. "
                f"Rapid file-share traffic burst across internal endpoints matches automated worm spreading "
                f"or ransomware staging behavior exploiting network share access."
            )
        elif attack_key == "infilteration":
            return (
                f"DNS-based infiltration from `{src}` to `{dst}:{dport}`: {size:,} bytes over {proto}. "
                f"Anomalously large DNS payload ({size} bytes, normal <512 bytes) indicates DNS tunneling "
                f"being used as a covert data exfiltration or lateral movement channel."
            )
        elif attack_key == "mitm":
            return (
                f"Man-in-the-middle interception indicator between `{src}` and `{dst}:{dport}`: "
                f"Inconsistent TTL hop counts and elevated retransmission bytes detected. "
                f"These indicators suggest ARP spoofing, rogue gateway injection, or SSL stripping activity "
                f"positioning an adversary between communicating endpoints."
            )
        elif attack_key == "shellcode":
            return (
                f"Shellcode staging attempt to `{dst}:{dport}` via {proto}: {size:,} bytes. "
                f"Connection to known exploit delivery port with anomalous binary payload structure "
                f"indicates in-memory shellcode injection or exploit framework staging."
            )
        elif attack_key == "password":
            return (
                f"Credential spraying / authentication brute-force targeting `{dst}:{dport}` via {proto}: "
                f"{flows_10s} rapid request bursts, {size:,} bytes. "
                f"Repeated authentication attempts with identical small request profiles indicate automated "
                f"credential stuffing or dictionary password spraying against user login endpoints."
            )
        elif attack_key == "fuzzers":
            return (
                f"Protocol fuzzing detected targeting `{dst}:{dport}` via {proto}: {size:,} bytes across {in_pkts:,} packets. "
                f"Non-RFC conformant payload structure and unexpected packet length variations match automated "
                f"fuzzer tools probing for memory corruption, buffer boundaries, or unhandled exceptions."
            )
        else:
            return (
                f"Multi-variate statistical anomaly on {proto} `{src}:{sport}` → `{dst}:{dport}`: "
                f"{size:,} bytes, {in_pkts:,} packets across {flows_10s} flows. "
                f"The flow's latent feature representation deviates significantly from the calibrated "
                f"benign manifold baseline across multiple NetFlow dimensions simultaneously."
            )

    def _explain_feature_deviation(self, feat: str, info: dict) -> str:
        """Generates dynamic explanations for why specific features drove the Autoencoder reconstruction error."""
        size = int(info.get("size", info.get("in_bytes", 0)))
        dport = int(info.get("dport", 0))
        out_bytes = int(info.get("out_bytes", 0))
        in_pkts = int(info.get("in_pkts", 1))
        duration_ms = int(info.get("duration", info.get("duration_ms", 0)))
        flows_10s = int(info.get("flows_10s", 1))

        feat_lower = feat.lower()

        if "num_pkts_up_to_128" in feat_lower:
            return (
                f"Disproportionate count of sub-128-byte packets detected. "
                f"Payload of {size:,} bytes across {in_pkts:,} packets indicates "
                f"TCP probe or SYN-only packets rather than normal data exchange."
            )
        elif "num_pkts_1024_to_1514" in feat_lower:
            return (
                f"High count of full-MTU (1024-1514 byte) packets: {in_pkts:,} total. "
                f"Sustained maximum-size packet transmission exceeding {size:,} bytes "
                f"is characteristic of volumetric flood or bulk data transfer."
            )
        elif "out_bytes" in feat_lower or "out_pkts" in feat_lower:
            ratio = size / max(out_bytes, 1)
            return (
                f"Extreme bidirectional byte asymmetry: {size:,} inbound vs {out_bytes:,} outbound. "
                f"A {ratio:.1f}:1 ingress-to-egress ratio significantly deviates from normal "
                f"request-response session baselines on port {dport}."
            )
        elif "in_bytes" in feat_lower or "longest_flow_pkt" in feat_lower:
            return (
                f"Payload volume of {size:,} bytes significantly deviates from the "
                f"calibrated normal distribution for port {dport} application traffic. "
                f"This dimension had the highest autoencoder reconstruction error in this flow."
            )
        elif "src_to_dst_avg_throughput" in feat_lower or "dst_to_src_avg_throughput" in feat_lower:
            throughput_mbps = (size * 8) / max(duration_ms, 1) / 1000 if duration_ms > 0 else 0
            return (
                f"Throughput of ~{throughput_mbps:.2f} Mbps deviates from benign baseline. "
                f"Abnormal wire-rate transfer indicates volumetric flood or bulk extraction activity."
            )
        elif "min_ttl" in feat_lower or "max_ttl" in feat_lower:
            return (
                "IP Time-To-Live hop count deviates from calibrated benign subnet baselines. "
                "Inconsistent TTL values indicate direct external probing, IP spoofing, or traffic relaying "
                "through an intermediate adversary-controlled node."
            )
        elif "duration" in feat_lower:
            return (
                f"Flow duration of {duration_ms}ms is abnormally {'short' if duration_ms < 500 else 'long'} "
                f"relative to the expected protocol lifecycle. "
                f"{'Very short durations indicate scan probes or SYN floods.' if duration_ms < 500 else 'Prolonged connections may indicate covert tunneling or interactive backdoor sessions.'}"
            )
        elif "unique_ports" in feat_lower:
            unique_ports = int(info.get("unique_ports", info.get("unique_ports_10s", 1)))
            return (
                f"Source IP targeted {unique_ports} distinct destination port(s) within a 10-second observation window. "
                f"Multi-port targeting from a single source is a primary indicator of port scanning or service enumeration."
            )
        elif "flows_10s" in feat_lower:
            return (
                f"Connection burst rate of {flows_10s} flows/10s from this source host significantly "
                f"exceeds normal session establishment rates. High burst density indicates automated "
                f"tooling, brute force attempts, or flood attack activity."
            )
        elif "tcp_flags" in feat_lower or "client_tcp_flags" in feat_lower:
            flags = int(info.get("flags", 0))
            flag_names = []
            if flags & 0x02: flag_names.append("SYN")
            if flags & 0x10: flag_names.append("ACK")
            if flags & 0x01: flag_names.append("FIN")
            if flags & 0x04: flag_names.append("RST")
            if flags & 0x08: flag_names.append("PSH")
            flag_str = "+".join(flag_names) if flag_names else f"0x{flags:02X}"
            return (
                f"TCP control flags ({flag_str}) in atypical combination or sequence. "
                f"Non-standard flag patterns indicate incomplete handshakes (SYN floods), "
                f"connection resets (port scans), or push-only data injections."
            )
        elif "retransmitted" in feat_lower:
            return (
                "Elevated retransmission byte/packet count detected. "
                "Unusually high retransmissions can indicate MITM interception, "
                "active packet dropping by adversary firewall, or intentional congestion attacks."
            )
        else:
            desc = self.feature_map.get(feat, feat.replace("_", " ").title())
            return (
                f"**{desc}** exhibited significant multivariate reconstruction error in the "
                f"Autoencoder latent space, deviating beyond the 99th-percentile benign manifold boundary."
            )

    def _generate_mitigation(self, attack_key: str, src: str, dst: str, dport: int, proto: str) -> str:
        """Generates targeted, actionable mitigation advice incorporating real endpoint facts."""
        if attack_key in ("scanning",):
            return (
                f"Apply border ACL/firewall rule to rate-limit or block `{src}` (max 10 new connections/s). "
                f"Verify port `{dport}` listening daemon suppresses version banners. "
                f"Enable port-knock or fail2ban on exposed administrative services."
            )
        elif attack_key == "reconnaissance":
            return (
                f"Block ICMP sweep and ARP discovery traffic from `{src}` at perimeter. "
                f"Implement internal VLAN micro-segmentation and deploy honeypot decoys. "
                f"Alert SOC team for immediate source IP investigation."
            )
        elif attack_key == "ddos":
            return (
                f"Activate upstream BGP Anycast traffic scrubbing for traffic destined to `{dst}:{dport}`. "
                f"Apply ingress rate-limiting on `{proto}` at the border router. "
                f"Enable TCP SYN cookies if targeting port `{dport}` and notify ISP for upstream filtering."
            )
        elif attack_key == "dos":
            return (
                f"Apply connection-rate throttling for `{src}` → `{dst}:{dport}` at firewall level. "
                f"Enable TCP SYN cookies and connection-drop timeouts on port `{dport}`. "
                f"If sustained, add `{src}` to block list and trigger incident escalation."
            )
        elif attack_key == "brute_force":
            return (
                f"Enforce Fail2Ban or IP lockout after 5 failed attempts for `{src}` on port `{dport}`. "
                f"Mandate multi-factor authentication (MFA) and key-based SSH authentication. "
                f"Disable password-based logins on port `{dport}` and rotate any credentials on `{dst}`."
            )
        elif attack_key == "theft":
            return (
                f"Immediately isolate endpoint `{src}` from network switch port. "
                f"Inspect egress firewall logs for sustained transfers to `{dst}`. "
                f"Apply DLP proxy filtering on all outbound traffic exceeding 100KB/flow. "
                f"Preserve forensic artifacts before quarantine."
            )
        elif attack_key in ("injection", "xss"):
            return (
                f"Inspect web application server logs on `{dst}:{dport}` for suspicious `{src}` request patterns. "
                f"Enforce parameterized database queries and escape all user inputs. "
                f"Update WAF rules to block SQL/XSS injection signatures from `{src}`."
            )
        elif attack_key == "exploits":
            return (
                f"Apply emergency CVE patch for any known vulnerability on service `{dst}:{dport}`. "
                f"Block `{src}` at perimeter firewall immediately. "
                f"Capture PCAP from the `{dst}:{dport}` session for forensic analysis."
            )
        elif attack_key in ("ransomware", "worms"):
            return (
                f"Quarantine host `{src}` from all internal subnets immediately. "
                f"Restrict SMB/RPC (port 445, 139) access across the domain. "
                f"Audit shared volumes on `{dst}` for unauthorized file modification or encryption activity. "
                f"Initiate snapshot restoration from last clean backup."
            )
        elif attack_key in ("bot", "backdoor"):
            return (
                f"Isolate host `{src}` from network. Terminate active socket to `{dst}:{dport}`. "
                f"Inspect `{src}` running processes for injected threads or malicious DLLs. "
                f"Scan memory with endpoint EDR tool and block outbound to `{dst}` at perimeter."
            )
        elif attack_key == "infilteration":
            return (
                f"Block anomalous DNS traffic from `{src}` at DNS resolver level. "
                f"Inspect all DNS requests to `{dst}` for tunneled payloads (data in QNAME/TXT fields). "
                f"Implement DNS query rate-limiting and log all high-volume DNS flows."
            )
        elif attack_key == "mitm":
            return (
                f"Enable Dynamic ARP Inspection (DAI) and DHCP snooping on local switch infrastructure. "
                f"Mandate end-to-end TLS certificate pinning for all `{dst}` connections. "
                f"Audit ARP table on gateway for spoofed MAC-to-IP bindings."
            )
        elif attack_key == "shellcode":
            return (
                f"Block inbound connections to port `{dport}` from `{src}` immediately. "
                f"Inspect `{dst}` process memory for injected shellcode segments. "
                f"Enable ASLR, DEP/NX on all services and audit endpoint execution logs."
            )
        elif attack_key == "fuzzers":
            return (
                f"Filter non-RFC-compliant {proto} packets on port `{dport}` at border gateway. "
                f"Monitor `{dst}` service daemon health for crash or memory fault indicators. "
                f"Implement strict packet length and flag validation at the application layer."
            )
        elif attack_key == "password":
            return (
                f"Implement progressive account lockout policy and CAPTCHA challenge on `{dst}:{dport}`. "
                f"Block offending source IP `{src}` at application firewall level. "
                f"Enforce multi-factor authentication (MFA) and audit failed authentication logs."
            )
        else:
            return (
                f"Inspect full packet PCAP between `{src}` and `{dst}:{dport}` to verify application integrity. "
                f"Cross-reference `{src}` against threat intelligence feeds. "
                f"Isolate if confirmed malicious and preserve forensic artifacts for SOC review."
            )

    def explain_anomaly_sync(self, packet_info: dict, hybrid_score: float, top_features: List[str] = None) -> str:
        """Generates comprehensive, dynamic, flow-grounded cybersecurity explanations."""
        attack_key = self._classify_anomaly(packet_info, top_features)
        context = (
            self.knowledge_base.get(attack_key)
            or self.knowledge_base.get(attack_key.lower())
            or self.knowledge_base.get("general_anomaly", {})
        )

        proto = str(packet_info.get("proto", "TCP")).upper()
        src = packet_info.get("src", "unknown")
        dst = packet_info.get("dst", "unknown")
        dport = int(packet_info.get("dport", 0))
        sport = int(packet_info.get("sport", 0))
        size = int(packet_info.get("size", packet_info.get("in_bytes", 0)))
        in_bytes = int(packet_info.get("in_bytes", size))
        out_bytes = int(packet_info.get("out_bytes", 0))
        in_pkts = int(packet_info.get("in_pkts", 1))
        duration_ms = int(packet_info.get("duration", packet_info.get("duration_ms", 0)))
        flows_10s = int(packet_info.get("flows_10s", 1))
        unique_ports = int(packet_info.get("unique_ports", packet_info.get("unique_ports_10s", 1)))

        attack_name = context.get("name", context.get("attack_type", attack_key.replace("_", " ").title()))
        category = str(context.get("category", attack_key)).upper()

        # Section 1: Header & Dynamic Threat Profile
        explanation = f"### 🚨 NetFlow Threat Detected: {attack_name}\n\n"
        explanation += (
            f"> **Threat Profile:** `{category}` | **Anomaly Score:** `{hybrid_score:.4f}` "
            f"*(Calibrated 99th Percentile Threshold Exceeded)*\n\n"
        )

        # Section 2: Observed Flow Telemetry
        explanation += "#### 📊 Observed Flow Telemetry\n"
        explanation += f"- **Endpoints:** `{src}:{sport}` ➔ `{dst}:{dport}` via **{proto}**\n"
        if out_bytes == 0:
            symmetry_str = "Unreplied / Half-Open Flow (no response bytes)"
        else:
            ratio = in_bytes / max(out_bytes, 1)
            symmetry_str = f"Bidirectional — {out_bytes:,} return bytes (ratio {ratio:.1f}:1)"
        explanation += f"- **Payload & Volume:** `{size:,} bytes` — {symmetry_str}\n"
        explanation += f"- **Packet Count:** `{in_pkts:,}` inbound packets\n"
        explanation += f"- **Temporal Context:** `{flows_10s}` flow(s) in 10s window | `{unique_ports}` unique destination port(s)\n"
        if duration_ms > 0:
            explanation += f"- **Connection Lifetime:** `{duration_ms:,} ms`\n"
        explanation += "\n"

        # Section 3: Grounded Behavioral Analysis
        explanation += "#### 🔍 Behavioral Analysis & Attack Mechanics\n"
        explanation += f"{self._generate_behavioral_narrative(attack_key, packet_info)}\n\n"

        # Section 4: Knowledge Base Context
        if context.get("description"):
            explanation += "#### 📚 Threat Intelligence Context\n"
            explanation += f"{context['description']}\n\n"
            if context.get("indicators"):
                explanation += "**Key Indicators of Compromise (IoC):**\n"
                for ind in context["indicators"][:3]:
                    explanation += f"- {ind}\n"
                explanation += "\n"

        # Section 5: XAI Feature Attribution
        if top_features:
            explanation += "#### 🧠 Explainable AI (XAI) — Feature Attribution\n"
            explanation += (
                "*The Autoencoder reconstruction error and Latent Isolation Forest identified the "
                "following flow dimensions as primary anomaly drivers:*\n\n"
            )
            for f in top_features:
                reason = self._explain_feature_deviation(f, packet_info)
                explanation += f"- **`{f}`**: {reason}\n"
            explanation += "\n"

        # Section 6: Contextual Mitigation
        explanation += "#### 🛡️ Recommended Incident Response & Mitigation\n"
        explanation += f"> {self._generate_mitigation(attack_key, src, dst, dport, proto)}\n\n"

        return explanation

    async def explain_anomaly(self, packet_info: dict, hybrid_score: float, top_features: List[str] = None):
        """Async wrapper for anomaly explanation."""
        import asyncio
        return await asyncio.to_thread(self.explain_anomaly_sync, packet_info, hybrid_score, top_features)
