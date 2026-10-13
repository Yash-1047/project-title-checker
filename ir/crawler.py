"""Small, safe crawler: fetches ONE user-supplied public page (no link following)."""
import ipaddress
import socket
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .extraction import extract_jsonld

USER_AGENT = "ProjectTitleCheckerIR/1.0 (educational crawler)"
ALLOWED_PORTS = {None, 80, 443}
MAX_BYTES = 1_000_000
MAX_REDIRECTS = 3


def validate_url(url):
    """Allow only public http(s) hosts on ports 80/443. Raises ValueError otherwise."""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a valid public http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not allowed.")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL.")
    if port not in ALLOWED_PORTS:
        raise ValueError("Only ports 80 and 443 are allowed.")
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ValueError("Host could not be resolved.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("Access to localhost, private or internal addresses is blocked.")
    return parsed.geturl()


def _get(url, timeout):
    """GET with manual redirects so every hop is re-validated; body size capped."""
    for _ in range(MAX_REDIRECTS + 1):
        url = validate_url(url)
        resp = requests.get(url, timeout=timeout, allow_redirects=False, stream=True,
                            headers={"User-Agent": USER_AGENT})
        if resp.is_redirect and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])
            resp.close()
            continue
        return url, resp
    raise ValueError("Too many redirects.")


def _allowed_by_robots(url, timeout):
    parsed = urlparse(url)
    robots_url = "%s://%s/robots.txt" % (parsed.scheme, parsed.netloc)
    try:
        _, resp = _get(robots_url, timeout)
        if resp.status_code != 200:
            return True
        rp = RobotFileParser()
        rp.parse(resp.raw.read(MAX_BYTES, decode_content=True).decode("utf-8", "ignore").splitlines())
        return rp.can_fetch(USER_AGENT, url)
    except (requests.RequestException, ValueError):
        return True


def readable_text(soup):
    for node in soup(["script", "style", "noscript", "template"]):
        node.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def parse_html(html, base_url=""):
    jsonld = extract_jsonld(html)
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else base_url
    links = []
    for a in soup.find_all("a", href=True):
        link = urljoin(base_url, a["href"])
        if urlparse(link).scheme in ("http", "https"):
            links.append(link.split("#")[0])
    return {"title": title, "text": readable_text(soup)[:10000],
            "links": list(dict.fromkeys(links))[:100], "jsonld": jsonld}


def crawl_page(url, timeout=6):
    final_url, resp = _get(url, timeout)
    if not _allowed_by_robots(final_url, timeout):
        resp.close()
        raise ValueError("The site's robots.txt disallows fetching this page.")
    try:
        resp.raise_for_status()
        if "html" not in resp.headers.get("Content-Type", "").lower():
            raise ValueError("The URL did not return an HTML page.")
        body = resp.raw.read(MAX_BYTES, decode_content=True)
    finally:
        resp.close()
    html = body.decode(resp.encoding or "utf-8", "ignore")
    result = parse_html(html, final_url)
    result["url"] = final_url
    return result
