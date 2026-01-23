"""Polymarket API authentication.

This module handles both levels of Polymarket authentication:

L1 Authentication (EIP-712):
- Used for deriving API credentials
- Uses private key to sign typed data
- One-time operation to generate API credentials

L2 Authentication (HMAC-SHA256):
- Used for all authenticated API requests
- Uses API key, secret, and passphrase
- Signs each request with HMAC
"""

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Optional

from eth_account import Account
from eth_account.messages import encode_typed_data

from ..core.exceptions import AuthenticationError
from ..config.models import SignatureType
from ..utils.logging import get_logger

logger = get_logger(__name__)


# EIP-712 domain for Polymarket CLOB authentication
CLOB_AUTH_DOMAIN = {
    "name": "ClobAuthDomain",
    "version": "1",
    "chainId": 137,  # Polygon mainnet
}

CLOB_AUTH_TYPES = {
    "ClobAuth": [
        {"name": "address", "type": "address"},
        {"name": "timestamp", "type": "string"},
        {"name": "nonce", "type": "uint256"},
        {"name": "message", "type": "string"},
    ],
}

CLOB_AUTH_MESSAGE = "This message attests that I control the given wallet"


@dataclass
class APICredentials:
    """Polymarket API credentials.

    These are derived from the private key using L1 authentication
    and used for L2 authentication on all subsequent requests.
    """

    api_key: str
    api_secret: str  # Base64 encoded
    passphrase: str

    def __repr__(self) -> str:
        """Hide sensitive data in repr."""
        return f"APICredentials(api_key='{self.api_key[:8]}...', secret='***', passphrase='***')"


@dataclass
class L1AuthHeaders:
    """Headers for L1 (EIP-712) authentication."""

    address: str
    signature: str
    timestamp: str
    nonce: int = 0

    def to_dict(self) -> dict[str, str]:
        """Convert to header dictionary."""
        return {
            "POLY_ADDRESS": self.address,
            "POLY_SIGNATURE": self.signature,
            "POLY_TIMESTAMP": self.timestamp,
            "POLY_NONCE": str(self.nonce),
        }


@dataclass
class L2AuthHeaders:
    """Headers for L2 (HMAC) authentication."""

    address: str
    signature: str
    timestamp: str
    api_key: str
    passphrase: str

    def to_dict(self) -> dict[str, str]:
        """Convert to header dictionary."""
        return {
            "POLY_ADDRESS": self.address,
            "POLY_SIGNATURE": self.signature,
            "POLY_TIMESTAMP": self.timestamp,
            "POLY_API_KEY": self.api_key,
            "POLY_PASSPHRASE": self.passphrase,
        }


class L1Authenticator:
    """L1 Authentication using EIP-712 signatures.

    Used for:
    - Creating API credentials
    - Deriving existing API credentials
    """

    def __init__(self, private_key: str):
        """Initialize with private key.

        Args:
            private_key: Hex-encoded private key (with or without 0x prefix)
        """
        # Normalize private key format
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self._private_key = private_key
        self._account = Account.from_key(private_key)
        self.address = self._account.address

    def sign_message(
        self,
        timestamp: Optional[str] = None,
        nonce: int = 0,
    ) -> L1AuthHeaders:
        """Create L1 authentication headers.

        Args:
            timestamp: Unix timestamp in milliseconds (current time if not provided)
            nonce: Nonce value (usually 0)

        Returns:
            L1AuthHeaders with signature
        """
        if timestamp is None:
            timestamp = str(int(time.time()))

        # Build EIP-712 typed data
        typed_data = {
            "types": CLOB_AUTH_TYPES,
            "primaryType": "ClobAuth",
            "domain": CLOB_AUTH_DOMAIN,
            "message": {
                "address": self.address,
                "timestamp": timestamp,
                "nonce": nonce,
                "message": CLOB_AUTH_MESSAGE,
            },
        }

        # Sign the typed data
        signable = encode_typed_data(full_message=typed_data)
        signed = self._account.sign_message(signable)

        return L1AuthHeaders(
            address=self.address,
            signature=signed.signature.hex(),
            timestamp=timestamp,
            nonce=nonce,
        )


class L2Authenticator:
    """L2 Authentication using HMAC-SHA256.

    Used for all authenticated API requests after credentials are obtained.
    """

    def __init__(
        self,
        credentials: APICredentials,
        address: str,
    ):
        """Initialize with API credentials.

        Args:
            credentials: API credentials (key, secret, passphrase)
            address: Wallet address for headers
        """
        self.credentials = credentials
        self.address = address

        # Decode the base64 secret
        try:
            self._secret = base64.b64decode(credentials.api_secret)
        except Exception as e:
            raise AuthenticationError(f"Invalid API secret format: {e}")

    def sign_request(
        self,
        method: str,
        path: str,
        body: str = "",
        timestamp: Optional[str] = None,
    ) -> L2AuthHeaders:
        """Create L2 authentication headers for a request.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: Request path (e.g., /order)
            body: Request body as string (empty for GET)
            timestamp: Unix timestamp in milliseconds (current if not provided)

        Returns:
            L2AuthHeaders with HMAC signature
        """
        if timestamp is None:
            timestamp = str(int(time.time()))

        # Build message to sign: timestamp + method + path + body
        message = timestamp + method.upper() + path + body

        # Create HMAC-SHA256 signature
        signature = hmac.new(
            self._secret,
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return L2AuthHeaders(
            address=self.address,
            signature=signature,
            timestamp=timestamp,
            api_key=self.credentials.api_key,
            passphrase=self.credentials.passphrase,
        )


class PolymarketAuth:
    """Combined authenticator for Polymarket API.

    Handles both L1 and L2 authentication, and credential management.
    """

    def __init__(
        self,
        private_key: str,
        credentials: Optional[APICredentials] = None,
        funder_address: Optional[str] = None,
        signature_type: SignatureType = SignatureType.GNOSIS_SAFE,
    ):
        """Initialize Polymarket authentication.

        Args:
            private_key: Hex-encoded private key
            credentials: Pre-existing API credentials (derived if not provided)
            funder_address: Proxy wallet address (if different from signer)
            signature_type: Type of wallet signature
        """
        self._l1 = L1Authenticator(private_key)
        self._l2: Optional[L2Authenticator] = None
        self.signature_type = signature_type

        # The funder is the proxy wallet that holds funds
        # The signer is the EOA that signs transactions
        self.signer_address = self._l1.address
        self.funder_address = funder_address or self.signer_address

        if credentials:
            self.set_credentials(credentials)

    def set_credentials(self, credentials: APICredentials) -> None:
        """Set API credentials for L2 authentication."""
        self._l2 = L2Authenticator(credentials, self.signer_address)
        logger.info("api_credentials_set", api_key=credentials.api_key[:8])

    @property
    def has_credentials(self) -> bool:
        """Check if L2 credentials are set."""
        return self._l2 is not None

    def get_l1_headers(self) -> dict[str, str]:
        """Get L1 authentication headers."""
        auth = self._l1.sign_message()
        return auth.to_dict()

    def get_l2_headers(
        self,
        method: str,
        path: str,
        body: str = "",
    ) -> dict[str, str]:
        """Get L2 authentication headers for a request.

        Args:
            method: HTTP method
            path: Request path
            body: Request body

        Returns:
            Dictionary of authentication headers

        Raises:
            AuthenticationError: If credentials not set
        """
        if not self._l2:
            raise AuthenticationError(
                "API credentials not set. Call set_credentials() first."
            )

        auth = self._l2.sign_request(method, path, body)
        return auth.to_dict()

    def get_websocket_auth(self) -> dict[str, str]:
        """Get authentication payload for WebSocket connection.

        Returns:
            Dictionary with apiKey, secret, and passphrase
        """
        if not self._l2:
            raise AuthenticationError(
                "API credentials not set for WebSocket auth."
            )

        return {
            "apiKey": self._l2.credentials.api_key,
            "secret": self._l2.credentials.api_secret,
            "passphrase": self._l2.credentials.passphrase,
        }


def load_credentials_from_env(
    api_key_env: str = "POLYBOT_API_KEY",
    api_secret_env: str = "POLYBOT_API_SECRET",
    api_passphrase_env: str = "POLYBOT_API_PASSPHRASE",
) -> Optional[APICredentials]:
    """Load API credentials from environment variables.

    Args:
        api_key_env: Environment variable name for API key
        api_secret_env: Environment variable name for API secret
        api_passphrase_env: Environment variable name for passphrase

    Returns:
        APICredentials if all values are set, None otherwise
    """
    import os

    api_key = os.getenv(api_key_env)
    api_secret = os.getenv(api_secret_env)
    passphrase = os.getenv(api_passphrase_env)

    if api_key and api_secret and passphrase:
        return APICredentials(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
        )

    return None


def load_private_key_from_env(
    private_key_env: str = "POLYBOT_PRIVATE_KEY",
) -> Optional[str]:
    """Load private key from environment variable.

    Args:
        private_key_env: Environment variable name

    Returns:
        Private key string if set, None otherwise
    """
    import os
    return os.getenv(private_key_env)
