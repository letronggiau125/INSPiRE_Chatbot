from flask import request, jsonify
from functools import wraps
import time
from collections import defaultdict
from typing import Dict, List, Optional
from config import Config

class RateLimiter:
    """Rate limiter implementation with burst allowance and IP whitelisting."""
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.whitelisted_ips: set = set()
        self.blacklisted_ips: set = set()
    
    def add_to_whitelist(self, ip: str) -> None:
        """Add an IP to the whitelist."""
        self.whitelisted_ips.add(ip)
        if ip in self.blacklisted_ips:
            self.blacklisted_ips.remove(ip)
    
    def add_to_blacklist(self, ip: str) -> None:
        """Add an IP to the blacklist."""
        self.blacklisted_ips.add(ip)
        if ip in self.whitelisted_ips:
            self.whitelisted_ips.remove(ip)
    
    def is_allowed(self, ip: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed from the IP.
        Returns (is_allowed, retry_after_seconds)
        """
        # Always allow whitelisted IPs
        if ip in self.whitelisted_ips:
            return True, None
            
        # Always block blacklisted IPs
        if ip in self.blacklisted_ips:
            return False, 3600  # 1 hour block
        
        now = time.time()
        minute_ago = now - 60
        
        # Remove old requests
        self.requests[ip] = [req_time for req_time in self.requests[ip] if req_time > minute_ago]
        
        # Check if under regular limit
        if len(self.requests[ip]) < self.requests_per_minute:
            self.requests[ip].append(now)
            return True, None
            
        # Check if burst limit is available
        if len(self.requests[ip]) < self.requests_per_minute + self.burst_limit:
            self.requests[ip].append(now)
            return True, None
            
        # Calculate retry after time
        oldest_request = min(self.requests[ip])
        retry_after = int(60 - (now - oldest_request))
        
        return False, retry_after

rate_limiter = RateLimiter(
    requests_per_minute=Config.RATE_LIMIT['requests_per_minute'],
    burst_limit=Config.RATE_LIMIT['burst_limit']
)

def rate_limit(f):
    """Decorator to apply rate limiting to routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        allowed, retry_after = rate_limiter.is_allowed(ip)
        
        if not allowed:
            response = {
                'error': Config.format_error_message('rate_limit_exceeded', timeout=retry_after),
                'retry_after': retry_after
            }
            return jsonify(response), 429
            
        return f(*args, **kwargs)
        
    return decorated_function 