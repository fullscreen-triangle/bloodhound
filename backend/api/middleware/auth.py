from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import json
import os

class AuthHandler:
    """Handle authentication and authorization"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.security = HTTPBearer()
        
    async def get_current_user(self, request: Request) -> Dict[str, Any]:
        """Get current user from request"""
        try:
            credentials: HTTPAuthorizationCredentials = await self.security(request)
            if not credentials:
                raise HTTPException(
                    status_code=401,
                    detail="Could not validate credentials"
                )
                
            token = credentials.credentials
            payload = self.decode_token(token)
            
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=401,
                    detail="Could not validate credentials"
                )
                
            return {"username": username, "permissions": payload.get("permissions", [])}
            
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
            
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode JWT token"""
        return jwt.decode(
            token,
            self.secret_key,
            algorithms=[self.algorithm]
        )
        
    def verify_permission(self, 
                         user: Dict[str, Any], 
                         required_permission: str) -> bool:
        """Verify user has required permission"""
        return required_permission in user.get("permissions", [])

class RoleBasedAuth:
    """Role-based access control"""
    
    def __init__(self, roles_file: Optional[Path] = None):
        self.roles_file = roles_file or Path("config/roles.json")
        self.roles = self._load_roles()
        
    def _load_roles(self) -> Dict[str, Any]:
        """Load role definitions from file"""
        try:
            with open(self.roles_file) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading roles: {str(e)}")
            return {}
            
    def has_permission(self, 
                      user_roles: list, 
                      required_permission: str) -> bool:
        """Check if user roles have required permission"""
        user_permissions = set()
        
        for role in user_roles:
            role_permissions = self.roles.get(role, {}).get("permissions", [])
            user_permissions.update(role_permissions)
            
        return required_permission in user_permissions

auth_handler = AuthHandler()
role_auth = RoleBasedAuth()

def require_permission(permission: str):
    """Decorator to require specific permission"""
    async def wrapper(request: Request):
        user = await auth_handler.get_current_user(request)
        if not role_auth.has_permission(user.get("roles", []), permission):
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions"
            )
        return True
    return wrapper
