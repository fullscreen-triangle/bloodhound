from typing import Dict, Any, Optional
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import logging
from pathlib import Path
import secrets
import bcrypt

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.secret_key = self.config.get('secret_key', secrets.token_urlsafe(32))
        self.algorithm = self.config.get('algorithm', "HS256")
        self.access_token_expire_minutes = self.config.get('access_token_expire_minutes', 30)
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logging.error(f"Password verification error: {str(e)}")
            return False
            
    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        try:
            return self.pwd_context.hash(password)
        except Exception as e:
            logging.error(f"Password hashing error: {str(e)}")
            raise
            
    def create_access_token(self, 
                           data: Dict[str, Any],
                           expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + (
                expires_delta or timedelta(minutes=self.access_token_expire_minutes)
            )
            to_encode.update({"exp": expire})
            encoded_jwt = jwt.encode(
                to_encode, 
                self.secret_key, 
                algorithm=self.algorithm
            )
            return encoded_jwt
        except Exception as e:
            logging.error(f"Token creation error: {str(e)}")
            raise
            
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
            
    def create_refresh_token(self, 
                           data: Dict[str, Any],
                           expires_delta: Optional[timedelta] = None) -> str:
        """Create refresh token"""
        try:
            to_encode = data.copy()
            expire = datetime.utcnow() + (
                expires_delta or timedelta(days=7)
            )
            to_encode.update({"exp": expire, "refresh": True})
            encoded_jwt = jwt.encode(
                to_encode, 
                self.secret_key, 
                algorithm=self.algorithm
            )
            return encoded_jwt
        except Exception as e:
            logging.error(f"Refresh token creation error: {str(e)}")
            raise
            
    def get_current_user(self, token: str = Security(oauth2_scheme)) -> Dict[str, Any]:
        """Get current user from token"""
        try:
            payload = self.verify_token(token)
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication credentials"
                )
            return {"user_id": user_id, "roles": payload.get("roles", [])}
        except Exception as e:
            logging.error(f"Current user retrieval error: {str(e)}")
            raise
