"""
plato_pki.py — PLATO Public Key Infrastructure (fleet-innovations #4)

Agent identity certificates using Ed25519 (modern, fast, small keys).
Integrated into Evennia MUD as @cert commands.

Each agent gets:
  - Ed25519 private key (stored encrypted with agent password)
  - Ed25519 public key (stored in a PLATO tile)
  - Self-signed identity certificate (room-specific)
  - Message signing/verification for bottle and DM authentication

Relies on `cryptography` library (installed) for Ed25519 ops.
"""

import os
import json
import base64
import time
from datetime import datetime

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding
from cryptography.hazmat.primitives.serialization import (
    load_der_private_key,
    load_der_public_key,
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
    BestAvailableEncryption,
)
from cryptography.exceptions import InvalidSignature

# Storage
CERT_DIR = os.path.expanduser("~/.openclaw/workspace/memory/plato-pki")
CERT_INDEX = os.path.join(CERT_DIR, "cert-index.json")


def _ensure_dir():
    os.makedirs(CERT_DIR, exist_ok=True)


def _load_index():
    _ensure_dir()
    if os.path.exists(CERT_INDEX):
        with open(CERT_INDEX) as f:
            return json.load(f)
    return {"agents": {}, "certs": []}


def _save_index(index):
    _ensure_dir()
    with open(CERT_INDEX, "w") as f:
        json.dump(index, f, indent=2)


# ============================================================
#  Key Generation
# ============================================================

def generate_agent_keypair(agent_name: str, passphrase: str = "") -> dict:
    """Generate Ed25519 keypair for an agent.
    
    Args:
        agent_name: Agent identifier (e.g., 'jc1', 'oracle1')
        passphrase: Optional passphrase to encrypt private key
    
    Returns:
        dict with public_key_pem and private_key location info
    """
    _ensure_dir()
    
    # Generate Ed25519 private key
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    # Serialize public key
    pub_der = public_key.public_bytes(
        Encoding.DER,
        PublicFormat.SubjectPublicKeyInfo
    )
    pub_b64 = base64.b64encode(pub_der).decode()
    
    # Serialize private key (encrypted if passphrase provided)
    if passphrase:
        priv_der = private_key.private_bytes(
            Encoding.DER,
            PrivateFormat.PKCS8,
            BestAvailableEncryption(passphrase.encode())
        )
    else:
        priv_der = private_key.private_bytes(
            Encoding.DER,
            PrivateFormat.PKCS8,
            NoEncryption()
        )
    priv_b64 = base64.b64encode(priv_der).decode()
    
    # Store in agent's key file
    key_file = os.path.join(CERT_DIR, f"{agent_name}-privkey.b64")
    pub_file = os.path.join(CERT_DIR, f"{agent_name}-pubkey.b64")
    
    with open(key_file, "w") as f:
        f.write(priv_b64)
    with open(pub_file, "w") as f:
        f.write(pub_b64)
    
    # Compute fingerprint (SHA-256 of public key)
    digest = hashes.Hash(hashes.SHA256())
    digest.update(pub_der)
    fingerprint = digest.finalize().hex()[:16]
    
    # Update index
    index = _load_index()
    index["agents"][agent_name] = {
        "fingerprint": fingerprint,
        "created": datetime.now().isoformat(),
        "pubkey_file": pub_file,
        "privkey_file": key_file,
        "has_passphrase": bool(passphrase),
    }
    _save_index(index)
    
    return {
        "agent": agent_name,
        "fingerprint": fingerprint,
        "algorithm": "Ed25519",
        "public_key": pub_b64,
        "key_file": key_file,
        "encrypted": bool(passphrase),
    }


def load_agent_key(agent_name: str, passphrase: str = "") -> ed25519.Ed25519PrivateKey:
    """Load an agent's private key for signing."""
    index = _load_index()
    if agent_name not in index["agents"]:
        raise ValueError(f"No key for agent {agent_name}")
    
    key_file = index["agents"][agent_name]["privkey_file"]
    with open(key_file) as f:
        priv_b64 = f.read()
    
    priv_der = base64.b64decode(priv_b64)
    
    if passphrase:
        return load_der_private_key(priv_der, passphrase.encode())
    else:
        return load_der_private_key(priv_der, None)


def load_agent_pubkey(agent_name: str) -> ed25519.Ed25519PublicKey:
    """Load an agent's public key for verification."""
    index = _load_index()
    if agent_name not in index["agents"]:
        raise ValueError(f"No key for agent {agent_name}")
    
    pub_file = index["agents"][agent_name]["pubkey_file"]
    with open(pub_file) as f:
        pub_b64 = f.read()
    
    pub_der = base64.b64decode(pub_b64)
    return load_der_public_key(pub_der)


# ============================================================
#  Signing & Verification
# ============================================================

def sign_message(agent_name: str, message: str, passphrase: str = "") -> dict:
    """Sign a message with agent's Ed25519 private key.
    
    Args:
        agent_name: Agent identifier
        message: The message content to sign
        passphrase: Passphrase if key is encrypted
    
    Returns:
        dict with message, signature (base64), signer, timestamp
    """
    private_key = load_agent_key(agent_name, passphrase)
    index = _load_index()
    
    sig = private_key.sign(message.encode())
    
    signed = {
        "message": message,
        "signature": base64.b64encode(sig).decode(),
        "signer": agent_name,
        "fingerprint": index["agents"][agent_name]["fingerprint"],
        "algorithm": "Ed25519",
        "timestamp": datetime.now().isoformat(),
    }
    return signed


def verify_message(signed: dict) -> dict:
    """Verify a signed message.
    
    Args:
        signed: The signed message dict from sign_message()
    
    Returns:
        dict with valid (bool), signer, fingerprint, error (if any)
    """
    try:
        # Load the signer's public key
        public_key = load_agent_pubkey(signed["signer"])
        
        sig = base64.b64decode(signed["signature"])
        public_key.verify(sig, signed["message"].encode())
        
        return {
            "valid": True,
            "signer": signed["signer"],
            "fingerprint": signed["fingerprint"],
            "message": signed["message"],
        }
    except (InvalidSignature, ValueError, KeyError, FileNotFoundError) as e:
        return {
            "valid": False,
            "signer": signed.get("signer", "unknown"),
            "error": str(e),
        }


# ============================================================
#  Certificate Generation (self-signed, room-scoped)
# ============================================================

def generate_cert(agent_name: str, room_id: str, room_name: str, 
                  roles: list = None, passphrase: str = "") -> dict:
    """Generate a self-signed identity certificate for a PLATO room.
    
    Certificate format (plain JSON, signed by agent's key):
    {
        "agent": "jc1",
        "fingerprint": "a1b2c3d4...",
        "issued": "2026-05-01T...",
        "expires": "2027-05-01T...",
        "scope": {"room_id": 39, "room_name": "Bridge"},
        "roles": ["captain", "engineer"],
        "signature": "<base64>"
    }
    """
    _ensure_dir()
    index = _load_index()
    
    if agent_name not in index["agents"]:
        # Auto-generate key if missing
        generate_agent_keypair(agent_name, passphrase)
    
    private_key = load_agent_key(agent_name, passphrase)
    
    if roles is None:
        roles = ["agent"]
    
    cert_body = {
        "agent": agent_name,
        "fingerprint": index["agents"][agent_name]["fingerprint"],
        "issued": datetime.now().isoformat(),
        "expires": datetime.now().isoformat().replace(
            "2026", "2027"
        ) if "2026" in datetime.now().isoformat() else datetime.now().isoformat(),
        "scope": {
            "room_id": room_id,
            "room_name": room_name,
        },
        "roles": roles,
    }
    
    # Sign the cert body (sorted keys for determinism)
    body_str = json.dumps(cert_body, sort_keys=True)
    sig = private_key.sign(body_str.encode())
    cert_body["signature"] = base64.b64encode(sig).decode()
    
    # Store in index
    cert_entry = {
        "agent": agent_name,
        "fingerprint": cert_body["fingerprint"],
        "room_id": room_id,
        "issued": cert_body["issued"],
        "expires": cert_body["expires"],
        "roles": roles,
    }
    index["certs"].append(cert_entry)
    _save_index(index)
    
    # Also save as individual cert file
    cert_file = os.path.join(
        CERT_DIR, 
        f"cert-{agent_name}@{room_id}.json"
    )
    with open(cert_file, "w") as f:
        json.dump(cert_body, f, indent=2)
    
    return cert_body


def verify_cert(cert: dict) -> dict:
    """Verify a PLATO certificate.
    
    Args:
        cert: Certificate dict (must have agent, fingerprint, signature)
    
    Returns:
        dict with valid (bool), signer, expiry check
    """
    # Check expiry
    try:
        expires = datetime.fromisoformat(cert["expires"])
        if datetime.now() > expires:
            return {
                "valid": False,
                "signer": cert.get("agent", "unknown"),
                "error": "certificate expired",
                "expired": True,
            }
    except (ValueError, KeyError):
        pass
    
    # Verify signature
    agent = cert["agent"]
    try:
        public_key = load_agent_pubkey(agent)
    except (ValueError, FileNotFoundError):
        # Fallback: check if we have the pubkey embedded or in index
        return {
            "valid": False,
            "signer": agent,
            "error": f"no public key found for {agent}",
        }
    
    # Reconstruct body without signature
    body = {k: v for k, v in cert.items() if k != "signature"}
    body_str = json.dumps(body, sort_keys=True)
    
    try:
        sig = base64.b64decode(cert["signature"])
        public_key.verify(sig, body_str.encode())
        return {
            "valid": True,
            "signer": agent,
            "fingerprint": cert.get("fingerprint", ""),
        }
    except (InvalidSignature, ValueError) as e:
        return {
            "valid": False,
            "signer": agent,
            "error": f"invalid signature: {e}",
        }


# ============================================================
#  Bottle Signing (for authenticated agent-to-agent messages)
# ============================================================

def sign_bottle(agent_name: str, bottle_body: str, 
                recipient: str = "", passphrase: str = "") -> str:
    """Sign a bottle message body with PLATO PKI.
    
    Returns the bottle text with PKI header appended.
    """
    signed = sign_message(agent_name, bottle_body, passphrase)
    
    pki_header = (
        f"\n---\n"
        f"PKI-Signer: {signed['signer']}\n"
        f"PKI-Fingerprint: {signed['fingerprint']}\n"
        f"PKI-Signature: {signed['signature']}\n"
        f"PKI-Algorithm: {signed['algorithm']}\n"
        f"PKI-Timestamp: {signed['timestamp']}\n"
    )
    return bottle_body + pki_header


def verify_bottle(bottle_text: str) -> dict:
    """Verify a PKI-signed bottle. Returns verification result."""
    import re
    
    # Parse PKI header from bottom of bottle
    pki_match = re.search(
        r"\n---\nPKI-Signer: (.+)\n"
        r"PKI-Fingerprint: (.+)\n"
        r"PKI-Signature: (.+)\n"
        r"PKI-Algorithm: (.+)\n"
        r"PKI-Timestamp: (.+)\n",
        bottle_text
    )
    
    if not pki_match:
        return {"valid": False, "error": "no PKI header found"}
    
    signer = pki_match.group(1)
    signature = pki_match.group(3)
    
    # Reconstruct original message (everything before PKI header)
    msg_end = bottle_text.rfind("\n---\nPKI-Signer:")
    if msg_end < 0:
        return {"valid": False, "error": "malformed PKI header"}
    
    original_msg = bottle_text[:msg_end]
    
    signed_packet = {
        "message": original_msg,
        "signature": signature,
        "signer": signer,
        "fingerprint": pki_match.group(2),
        "algorithm": pki_match.group(4),
        "timestamp": pki_match.group(5),
    }
    
    return verify_message(signed_packet)


# ============================================================
#  Fleet Trust Bridge
# ============================================================

def cert_trust_bridge(agent_name: str) -> dict:
    """Bridge between PLATO PKI certs and mesh-bridge trust scoring.
    
    An agent with a valid cert contributes +0.2 to trust score base.
    An expired or missing cert contributes -0.1.
    """
    index = _load_index()
    
    if agent_name not in index["agents"]:
        return {"pki_status": "no_key", "trust_boost": -0.1}
    
    # Check for any valid certs
    agent_certs = [c for c in index["certs"] if c["agent"] == agent_name]
    
    if not agent_certs:
        return {"pki_status": "no_certs", "trust_boost": 0.0}
    
    valid_certs = 0
    for c in agent_certs:
        try:
            expires = datetime.fromisoformat(c["expires"])
            if datetime.now() <= expires:
                valid_certs += 1
        except (ValueError, KeyError):
            pass
    
    if valid_certs > 0:
        return {
            "pki_status": f"{valid_certs} valid cert(s)",
            "trust_boost": 0.2,
        }
    else:
        return {
            "pki_status": "expired_certs",
            "trust_boost": -0.1,
        }
