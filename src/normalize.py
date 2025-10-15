# waf/src/normalize.py
import re
import urllib.parse

# Basic normalization patterns
_RE_UUID = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b')
_RE_ISO_TS = re.compile(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b')
_RE_NUM = re.compile(r'\b\d+\b')
_RE_HEX = re.compile(r'\b0x[0-9a-fA-F]+\b')
_RE_BASE64 = re.compile(r'\b(?:[A-Za-z0-9+/]{40,}={0,2})\b')
_RE_IP = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_RE_EMAIL = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b')
_RE_TOKEN_PARAM = re.compile(r'(?i)(?:token|session|auth|jwt|access_token|api_key)=([^&\s]+)')

def _normalize_component(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = s.lower()
    s = _RE_UUID.sub("<UUID>", s)
    s = _RE_ISO_TS.sub("<TS>", s)
    s = _RE_IP.sub("<IP>", s)
    s = _RE_EMAIL.sub("<EMAIL>", s)
    s = _RE_HEX.sub("<HEX>", s)
    s = _RE_BASE64.sub("<BLOB>", s)
    s = _RE_TOKEN_PARAM.sub(lambda m: m.group(0).split("=")[0] + "=<TOKEN>", s)
    s = _RE_NUM.sub("<NUM>", s)
    s = re.sub(r'/+', '/', s)
    s = re.sub(r'[^a-z0-9/<>=_\-.\?:&%]+', '<SYM>', s)
    s = re.sub(r'(<NUM>)(?:\s*\1)+', r'\1', s)
    return s

def normalize_request(method: str, uri: str, user_agent: str = "") -> str:
    try:
        parsed = urllib.parse.urlparse(uri)
        path = parsed.path or "/"
        query = parsed.query or ""
        path_norm = _normalize_component(path)
        if query:
            params = urllib.parse.parse_qsl(query, keep_blank_values=True)
            norm_params = []
            for k, v in params:
                k_n = _normalize_component(k)
                v_n = _normalize_component(v)
                norm_params.append(f"{k_n}={v_n}")
            query_norm = "&".join(norm_params)
            full = f"{method} {path_norm}?{query_norm}"
        else:
            full = f"{method} {path_norm}"
    except Exception:
        full = _normalize_component(f"{method} {uri}")

    ua_norm = _normalize_component(user_agent) if user_agent else ""
    return f"{full} user-agent={ua_norm}" if ua_norm else full

# Apache/Nginx combined log parse
_APACHE_COMBINED_RE = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<request>[^"]+)" (?P<status>\d{3}) (?P<size>\S+) "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)"'
)

def parse_access_log_line(line: str):
    m = _APACHE_COMBINED_RE.match(line)
    if not m:
        rr = re.search(r'\"([A-Z]+) ([^"]+) HTTP/[\d.]+"', line)
        ua = ""
        ua_m = re.search(r'\" ([^"]*)\"$|\"[^"]*\" \"([^"]*)\"$', line)
        if ua_m:
            ua = (ua_m.group(1) or ua_m.group(2) or "")
        if rr:
            return rr.group(1), rr.group(2), ua
        return None
    req = m.group("request")
    agent = m.group("agent")
    parts = req.split()
    if len(parts) >= 2:
        method = parts[0]
        uri = parts[1]
        return method, uri, agent
    return None
