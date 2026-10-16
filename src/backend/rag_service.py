import httpx
import os
import json
import re
from typing import List, Dict, Any

class RAGExplanationService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-placeholder")
        
        # Load comprehensive knowledge base dynamically
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        self._load_knowledge_base()

        # Build comprehensive feature map covering all 116 dataset features
        self.feature_map = self._build_feature_map()

        # Initialize vector index for RAG semantic retrieval
        self.kb_index = None
        try:
            from .kb_index import KBIndex
            kb_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'knowledge_base_enhanced.json')
            kb_file = os.path.abspath(kb_file)
            if not os.path.exists(kb_file):
                kb_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'knowledge_base.json')
                kb_file = os.path.abspath(kb_file)

            self.kb_index = KBIndex(kb_file)
            try:
                self.kb_index.load()
            except Exception:
                try:
                    self.kb_index.build(overwrite=True)
                except Exception:
                    self.kb_index = None
        except Exception:
            self.kb_index = None

    def _load_knowledge_base(self):
        """Loads and indexes knowledge base entries from JSON into dynamic lookup dictionaries."""
        kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'knowledge_base_enhanced.json')
        kb_path = os.path.abspath(kb_path)
        if not os.path.exists(kb_path):
            kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'knowledge_base.json')
            kb_path = os.path.abspath(kb_path)

        raw_entries = []
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    raw_entries = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load knowledge base JSON: {e}")

        # Index by attack_type, name slug, and signature keywords
        for entry in raw_entries:
            name = entry.get('name', '')
            attack_type = str(entry.get('attack_type', '')).lower()
            slug = re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))
            
            item = {
                "name": name,
                "category": attack_type,
                "description": entry.get('description', f"Network threat in category {attack_type.upper()}."),
                "details": f"Signature: {name}. Category: {attack_type.upper()}.",
                "indicators": entry.get('indicators', ["Anomaly score threshold exceeded"]),
                "feature_explanation": entry.get('feature_explanation', {})
            }

            self.knowledge_base[slug] = item
            if attack_type not in self.knowledge_base:
                self.knowledge_base[attack_type] = item

        # Ensure fallback defaults exist
        fallbacks = {
            "dos": {
                "name": "Denial of Service (DoS)",
                "category": "dos",
                "description": "Denial of Service (DoS) attack.",
                "details": "Attempts to exhaust system or network resources (e.g. SYN Flooding, Smurf ICMP amplification, Teardrop fragmentation).",
                "indicators": ["High serror_rate", "High count in short window", "Large src_bytes"]
            },
            "probe": {
                "name": "Probing / Network Reconnaissance",
                "category": "probe",
                "description": "Probing or Surveillance attack.",
                "details": "Scans network ports or IP addresses to discover open services and vulnerabilities (e.g. Nmap, Portsweep, IPSweep, Satan).",
                "indicators": ["High diff_srv_rate", "Multiple services targeted", "High connection rejection rate (REJ)"]
            },
            "r2l": {
                "name": "Remote-to-Local (R2L) Unauthorized Access",
                "category": "r2l",
                "description": "Remote to Local (R2L) attack.",
                "details": "Attacker sends packets over the network to exploit vulnerabilities and gain unauthorized user account access (e.g. Brute Force Password Guessing, FTP Write, IMAP exploit).",
                "indicators": ["Multiple failed logins (num_failed_logins)", "Unauthenticated access on FTP/Telnet/SSH", "Guest login attempts"]
            },
            "u2r": {
                "name": "User-to-Root (U2R) Privilege Escalation",
                "category": "u2r",
                "description": "User to Root (U2R) attack.",
                "details": "Attacker starts with normal user access and exploits a local system vulnerability to gain root/administrator privileges (e.g. Buffer Overflow, Rootkit, Loadmodule, Setuid Perl).",
                "indicators": ["Root shell spawned (root_shell = 1)", "Su attempts to root", "System file modifications"]
            },
            "general_anomaly": {
                "name": "Statistical Network Anomaly",
                "category": "general_anomaly",
                "description": "Statistical Network Anomaly.",
                "details": "The hybrid autoencoder + isolation forest model detected a major reconstruction error or multi-dimensional outlier deviating from normal baseline traffic.",
                "indicators": ["High combined hybrid score", "Unusual combination of TCP flags and byte ratios"]
            }
        }
        for k, v in fallbacks.items():
            if k not in self.knowledge_base:
                self.knowledge_base[k] = v

    def _build_feature_map(self) -> Dict[str, str]:
        """Provides human-readable cybersecurity explanations for all dataset features."""
        return {
            # Continuous KDD Features
            "duration": "Length of the connection in seconds",
            "src_bytes": "Data bytes sent from source to destination",
            "dst_bytes": "Data bytes sent from destination to source",
            "land": "Land attack indicator (Source IP and Port match Destination IP and Port)",
            "wrong_fragment": "Number of wrong/corrupted IP packet fragments",
            "urgent": "Number of urgent TCP packets",
            "hot": "Number of 'hot' indicator actions (accessing system directories/executables)",
            "num_failed_logins": "Number of failed authentication/login attempts",
            "logged_in": "Successful login status (1 = logged in, 0 = unauthenticated)",
            "num_compromised": "Number of compromised system states detected",
            "root_shell": "Root shell privilege obtained (1 = root shell active)",
            "su_attempted": "Attempted to switch user to root ('su root')",
            "num_root": "Number of root-level operations performed",
            "num_file_creations": "Number of new files created on the system",
            "num_shells": "Number of interactive command shell prompts opened",
            "num_access_files": "Number of operations on control/security files",
            "num_outbound_cmds": "Number of outbound command calls in an FTP session",
            "is_host_login": "Host administrative login status",
            "is_guest_login": "Guest account login status",
            "count": "Number of connections to the same destination host in past 2 seconds",
            "srv_count": "Number of connections to the same service port in past 2 seconds",
            "serror_rate": "% of connections that activated SYN error flags (S0/S1/S2/S3)",
            "srv_serror_rate": "% of service connections that activated SYN error flags",
            "rerror_rate": "% of connections that activated REJ (rejection) flags",
            "srv_rerror_rate": "% of service connections that activated REJ flags",
            "same_srv_rate": "% of connections to the same service",
            "diff_srv_rate": "% of connections to different services (indicator of port scanning)",
            "srv_diff_host_rate": "% of connections to different hosts on the same service (indicator of IP sweeping)",
            "dst_host_count": "Destination host count in cumulative 100-connection window",
            "dst_host_srv_count": "Destination host service count in 100-connection window",
            "dst_host_same_srv_rate": "% of destination host connections to the same service",
            "dst_host_diff_srv_rate": "% of destination host connections to different services",
            "dst_host_same_src_port_rate": "% of destination host connections from the same source port",
            "dst_host_srv_diff_host_rate": "% of destination host connections to different hosts for the service",
            "dst_host_serror_rate": "Destination host SYN error rate",
            "dst_host_srv_serror_rate": "Destination host service SYN error rate",
            "dst_host_rerror_rate": "Destination host connection rejection rate",
            "dst_host_srv_rerror_rate": "Destination host service connection rejection rate",
            
            # Protocols
            "protocol_type_icmp": "ICMP network control message protocol traffic",
            "protocol_type_tcp": "TCP connection-oriented protocol traffic",
            "protocol_type_udp": "UDP connectionless datagram protocol traffic",

            # Flags
            "flag_S0": "SYN sent, no ACK received (unanswered connection attempt / SYN flood)",
            "flag_S1": "SYN sent and acknowledged, but connection not established",
            "flag_S2": "Connection established, but close request not acknowledged",
            "flag_S3": "Connection attempted with reset error",
            "flag_SF": "Normal connection establishment and termination (SYN-ACK-FIN)",
            "flag_REJ": "Connection attempt rejected by target firewall or closed port",
            "flag_RSTO": "Connection reset by originator",
            "flag_RSTR": "Connection reset by responder",
            "flag_RSTOS0": "Reset by originator with SYN error",
            "flag_SH": "SYN sent with FIN flag set (stealth scan attempt)",
            "flag_OTH": "Other non-standard TCP flag state",

            # Key Services
            "service_http": "HTTP web server traffic (Port 80)",
            "service_http_443": "HTTPS secure web server traffic (Port 443)",
            "service_ftp": "FTP control command connection (Port 21)",
            "service_ftp_data": "FTP data transfer stream (Port 20)",
            "service_telnet": "Telnet unencrypted remote terminal session (Port 23)",
            "service_ssh": "SSH secure shell remote login (Port 22)",
            "service_smtp": "SMTP mail transfer traffic (Port 25)",
            "service_domain": "DNS domain name resolution service (Port 53)",
            "service_domain_u": "UDP DNS resolution service",
            "service_eco_i": "ICMP echo request (ping)",
            "service_ecr_i": "ICMP echo reply",
            "service_private": "Private / non-standard administrative port traffic",
            "service_auth": "Ident / Auth authentication service (Port 113)",
            "service_exec": "Remote execution daemon (rexec / Port 512)",
            "service_login": "Remote login daemon (rlogin / Port 513)",
            "service_shell": "Remote shell execution (rsh / Port 514)",
            "service_finger": "Finger user information service (Port 79)",
            "service_imap4": "IMAP4 mail client access (Port 143)",
            "service_pop_3": "POP3 mail retrieving client (Port 110)",
            "service_sql_net": "Oracle SQL*Net database traffic",
            "service_IRC": "Internet Relay Chat protocol (Port 6667)"
        }

    def explain_anomaly_sync(self, packet_info: dict, hybrid_score: float, top_features: List[str] = None) -> str:
        """Generates structured cybersecurity explanations for detected network anomalies."""
        attack_key = self._classify_anomaly(packet_info, top_features)
        context = self.knowledge_base.get(attack_key, self.knowledge_base.get("general_anomaly"))

        proto = str(packet_info.get('proto', 'unknown')).upper()
        if proto == '0': proto = 'UNKNOWN'

        explanation = f"### 🚨 Anomaly Detected: {context.get('name', 'Network Threat')}\n\n"
        explanation += f"> **Technical Summary:** A packet from **{packet_info.get('src', 'unknown')}** to **{packet_info.get('dst', 'unknown')}** using **{proto}** (Service: `{packet_info.get('service', 'unknown')}`) was flagged with a high hybrid anomaly score of **{hybrid_score:.4f}**.\n\n"
        
        explanation += f"#### Analysis & Threat Profile\n{context.get('description', '')}\n\n"
        if 'details' in context:
            explanation += f"**Detailed Mechanics:** {context['details']}\n\n"
        
        explanation += f"#### Key Attack Indicators\n"
        for ind in context.get('indicators', []):
            explanation += f"- {ind}\n"
        explanation += "\n"
        
        if top_features:
            explanation += f"#### XAI Top Contributing Factors (SHAP + AE Reconstruction)\n"
            for f in top_features:
                readable = self.feature_map.get(f)
                if not readable:
                    if f.startswith("service_"):
                        readable = f"Targeting specific service daemon: `{f.replace('service_', '')}`"
                    elif f.startswith("flag_"):
                        readable = f"TCP state flag activated: `{f.replace('flag_', '')}`"
                    elif f.startswith("protocol_type_"):
                        readable = f"Transport layer protocol: `{f.replace('protocol_type_', '').upper()}`"
                    else:
                        readable = f.replace('_', ' ').title()
                explanation += f"- **{f}**: {readable}\n"
            explanation += "\n"

        if self.kb_index is not None:
            try:
                if top_features:
                    query_text = ' '.join(top_features) + f" {attack_key} {context.get('name', '')}"
                else:
                    q_tokens: List[str] = []
                    for k in ('service', 'proto', 'src_bytes', 'dst_host_count', 'diff_srv_rate', 'num_failed_logins', 'serror_rate'):
                        if k in packet_info:
                            q_tokens.append(f"{k}:{packet_info[k]}")
                    q_tokens.append(attack_key)
                    query_text = ' '.join(map(str, q_tokens))
                    
                hits = self.kb_index.query(query_text, top_k=3)
                if hits:
                    explanation += "#### 📚 Knowledge Base Vector References\n"
                    seen_names = set()
                    for ent, score in hits:
                        name = ent.get('name', 'Unknown')
                        if name in seen_names:
                            continue
                        seen_names.add(name)
                        explanation += f"- **{name}** (Category: `{ent.get('attack_type', 'unknown').upper()}`): {ent.get('description', '')} _[Similarity Score: {score:.2f}]_\n"
            except Exception as e:
                explanation += f"\n*(Notice: Knowledge Base index lookup notice - {str(e)})*\n"

        return explanation

    async def explain_anomaly(self, packet_info: dict, hybrid_score: float, top_features: List[str] = None):
        return self.explain_anomaly_sync(packet_info, hybrid_score, top_features)

    def _classify_anomaly(self, info: dict, top_features: List[str] = None) -> str:
        """
        Multi-tier anomaly classification engine matching specific attack signatures
        first, then falling back to high-level super-classes (DoS, Probe, R2L, U2R).
        """
        features_str = ' '.join(top_features).lower() if top_features else ''
        service = str(info.get('service', '')).lower()
        land = info.get('land', 0)
        serror_rate = info.get('serror_rate', 0.0)
        diff_srv_rate = info.get('diff_srv_rate', 0.0)
        src_bytes = info.get('src_bytes', 0)
        count = info.get('count', 0)
        proto = str(info.get('proto', '')).lower()

        # 1. SPECIFIC SIGNATURE MATCHING (Tier 1)
        # Land Attack
        if land == 1 or 'land' in features_str:
            return "land_attack"

        # Teardrop / Ping of Death
        if 'wrong_fragment' in features_str or (proto == 'icmp' and src_bytes > 60000):
            return "teardrop_attack" if 'wrong_fragment' in features_str else "pod_ping_of_death"

        # Smurf ICMP Flood
        if proto == 'icmp' and src_bytes > 1000 and count > 20:
            return "smurf_attack"

        # Neptune SYN Flood
        if 'flag_s0' in features_str or 'serror_rate' in features_str or serror_rate > 0.6:
            return "neptune_syn_flood"

        # Password Guessing / Brute Force (R2L)
        if any(k in features_str for k in ['num_failed_logins', 'is_guest_login']) or info.get('num_failed_logins', 0) > 0:
            return "password_guessing_guess_passwd_brute_force"

        # Buffer Overflow / Privilege Escalation (U2R)
        if any(k in features_str for k in ['root_shell', 'su_attempted', 'num_root', 'num_file_creations']) or info.get('root_shell', 0) == 1:
            return "buffer_overflow_user_to_root"

        # Port Scanning / Probing (Nmap, Satan, Portsweep, IPSweep)
        # Require sufficient connection count (count >= 5) to distinguish true scanning from browsing 2 web pages
        if count >= 5 and ('diff_srv_rate' in features_str or diff_srv_rate > 0.5 or 'srv_diff_host_rate' in features_str):
            if 'dst_host_count' in features_str or info.get('dst_host_count', 0) > 20:
                return "nmap_port_scan"
            return "portsweep_attack"

        # 2. CATEGORY FALLBACK MATCHING (Tier 2)
        if any(k in features_str for k in ['count', 'srv_count', 'serror_rate', 'src_bytes']):
            return "dos"
        if any(k in features_str for k in ['diff_srv_rate', 'dst_host_count', 'srv_diff_host_rate']) and count >= 5:
            return "probe"
        if any(k in features_str for k in ['num_failed_logins', 'is_guest_login']):
            return "r2l"
        if any(k in features_str for k in ['root_shell', 'su_attempted', 'num_root']):
            return "u2r"

        # 3. HEURISTIC FALLBACK (Tier 3)
        if land == 1 or (serror_rate > 0.5 and count >= 5) or src_bytes > 50000 or count > 50:
            return "dos"
        if count >= 10 and diff_srv_rate > 0.6:
            return "probe"
        if info.get('num_failed_logins', 0) > 1:
            return "r2l"
        if info.get('root_shell', 0) == 1:
            return "u2r"

        return "general_anomaly"
