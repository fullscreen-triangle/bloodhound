from typing import Dict, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64
import os
import logging
from pathlib import Path

class EncryptionManager:
    """Data encryption and decryption manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.key = self._generate_or_load_key()
        self.fernet = Fernet(self.key)
        self._init_asymmetric_keys()
        
    def encrypt_data(self, 
                    data: Union[str, bytes],
                    method: str = 'symmetric') -> bytes:
        """Encrypt data using specified method"""
        try:
            if isinstance(data, str):
                data = data.encode()
                
            if method == 'symmetric':
                return self.fernet.encrypt(data)
            elif method == 'asymmetric':
                return self.public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            else:
                raise ValueError(f"Unknown encryption method: {method}")
                
        except Exception as e:
            logging.error(f"Encryption error: {str(e)}")
            raise
            
    def decrypt_data(self, 
                    encrypted_data: bytes,
                    method: str = 'symmetric') -> bytes:
        """Decrypt data using specified method"""
        try:
            if method == 'symmetric':
                return self.fernet.decrypt(encrypted_data)
            elif method == 'asymmetric':
                return self.private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            else:
                raise ValueError(f"Unknown decryption method: {method}")
                
        except Exception as e:
            logging.error(f"Decryption error: {str(e)}")
            raise
            
    def generate_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password"""
        try:
            salt = salt or os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key
            
        except Exception as e:
            logging.error(f"Key generation error: {str(e)}")
            raise
            
    def _generate_or_load_key(self) -> bytes:
        """Generate new key or load existing one"""
        key_path = self.config.get('key_path', 'data/encryption/key.key')
        try:
            if Path(key_path).exists():
                with open(key_path, 'rb') as key_file:
                    return key_file.read()
            else:
                key = Fernet.generate_key()
                Path(key_path).parent.mkdir(parents=True, exist_ok=True)
                with open(key_path, 'wb') as key_file:
                    key_file.write(key)
                return key
                
        except Exception as e:
            logging.error(f"Key loading error: {str(e)}")
            raise
            
    def _init_asymmetric_keys(self):
        """Initialize RSA key pair"""
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.private_key = private_key
            self.public_key = private_key.public_key()
            
        except Exception as e:
            logging.error(f"Asymmetric key initialization error: {str(e)}")
            raise
    
    def export_public_key(self) -> bytes:
        """Export public key in PEM format"""
        try:
            return self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        except Exception as e:
            logging.error(f"Public key export error: {str(e)}")
            raise
