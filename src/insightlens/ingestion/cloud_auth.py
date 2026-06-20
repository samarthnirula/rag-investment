"""OAuth2 helpers for connecting cloud storage accounts.

Each function returns an authorization URL to redirect the user to.
After the user authorizes, the provider redirects back to the given callback URL
with a short-lived authorization code. Exchange that code for a long-lived
refresh token server-side and store it in the per-user credentials table.

Env vars required (set in .env):
  GOOGLE_DRIVE_CLIENT_ID / GOOGLE_DRIVE_CLIENT_SECRET
  DROPBOX_APP_KEY / DROPBOX_APP_SECRET
  ONEDRIVE_CLIENT_ID / ONEDRIVE_CLIENT_SECRET
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import requests


# ── Helpers ────────────────────────────────────────────────────────────────────

@dataclass
class OAuthTokens:
    access_token: str
    refresh_token: Optional[str]
    expires_in: int          # seconds until access_token expires
    provider: str


def _env(key: str, fallback: str = "") -> str:
    return os.getenv(key, fallback)


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _generate_state() -> str:
    return secrets.token_urlsafe(32)


# ── Google Drive ───────────────────────────────────────────────────────────────

_GOOGLE_DRIVE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_DRIVE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def google_drive_auth_url(callback_url: str, state: str) -> str:
    client_id = _env("GOOGLE_DRIVE_CLIENT_ID")
    if not client_id:
        raise ValueError("GOOGLE_DRIVE_CLIENT_ID is not set in .env")
    return (
        f"{_GOOGLE_DRIVE_AUTH_URL}"
        f"?client_id={client_id}"
        f"&redirect_uri={urllib.parse.quote(callback_url)}"
        f"&response_type=code"
        f"&scope=https://www.googleapis.com/auth/drive.readonly"
        f"&access_type=offline"
        f"&prompt=consent"
        f"&state={state}"
    )


def exchange_google_drive_code(code: str, callback_url: str) -> OAuthTokens:
    client_id = _env("GOOGLE_DRIVE_CLIENT_ID")
    client_secret = _env("GOOGLE_DRIVE_CLIENT_SECRET")
    resp = requests.post(
        _GOOGLE_DRIVE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": callback_url,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return OAuthTokens(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token"),
        expires_in=int(data.get("expires_in", 3600)),
        provider="google_drive",
    )


def refresh_google_drive_token(refresh_token: str) -> OAuthTokens:
    client_id = _env("GOOGLE_DRIVE_CLIENT_ID")
    client_secret = _env("GOOGLE_DRIVE_CLIENT_SECRET")
    resp = requests.post(
        _GOOGLE_DRIVE_TOKEN_URL,
        data={
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return OAuthTokens(
        access_token=data["access_token"],
        refresh_token=refresh_token,
        expires_in=int(data.get("expires_in", 3600)),
        provider="google_drive",
    )


# ── Dropbox ────────────────────────────────────────────────────────────────────

_DROPBOX_AUTH_URL = "https://www.dropbox.com/oauth2/authorize"
_DROPBOX_TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"


def dropbox_auth_url(callback_url: str, state: str) -> str:
    app_key = _env("DROPBOX_APP_KEY")
    if not app_key:
        raise ValueError("DROPBOX_APP_KEY is not set in .env")
    return (
        f"{_DROPBOX_AUTH_URL}"
        f"?client_id={app_key}"
        f"&redirect_uri={urllib.parse.quote(callback_url)}"
        f"&response_type=code"
        f"&state={state}"
    )


def exchange_dropbox_code(code: str, callback_url: str) -> OAuthTokens:
    app_key = _env("DROPBOX_APP_KEY")
    app_secret = _env("DROPBOX_APP_SECRET")
    resp = requests.post(
        _DROPBOX_TOKEN_URL,
        data={
            "code": code,
            "client_id": app_key,
            "client_secret": app_secret,
            "redirect_uri": callback_url,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return OAuthTokens(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token"),
        expires_in=int(data.get("expires_in", 14400)),
        provider="dropbox",
    )


def refresh_dropbox_token(refresh_token: str) -> OAuthTokens:
    app_key = _env("DROPBOX_APP_KEY")
    app_secret = _env("DROPBOX_APP_SECRET")
    resp = requests.post(
        _DROPBOX_TOKEN_URL,
        data={
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return OAuthTokens(
        access_token=data["access_token"],
        refresh_token=refresh_token,
        expires_in=int(data.get("expires_in", 14400)),
        provider="dropbox",
    )


# ── OneDrive ────────────────────────────────────────────────────────────────────

_ONEDRIVE_AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
_ONEDRIVE_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"


def onedrive_auth_url(callback_url: str, state: str) -> str:
    client_id = _env("ONEDRIVE_CLIENT_ID")
    if not client_id:
        raise ValueError("ONEDRIVE_CLIENT_ID is not set in .env")
    return (
        f"{_ONEDRIVE_AUTH_URL}"
        f"?client_id={client_id}"
        f"&redirect_uri={urllib.parse.quote(callback_url)}"
        f"&response_type=code"
        f"&scope=https://graph.microsoft.com/Files.Read.All https://graph.microsoft.com/Sites.Read.All"
        f"&state={state}"
    )


def exchange_onedrive_code(code: str, callback_url: str) -> OAuthTokens:
    client_id = _env("ONEDRIVE_CLIENT_ID")
    client_secret = _env("ONEDRIVE_CLIENT_SECRET")
    resp = requests.post(
        _ONEDRIVE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": callback_url,
            "grant_type": "authorization_code",
            "scope": "https://graph.microsoft.com/Files.Read.All https://graph.microsoft.com/Sites.Read.All",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return OAuthTokens(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token"),
        expires_in=int(data.get("expires_in", 3600)),
        provider="onedrive",
    )


def refresh_onedrive_token(refresh_token: str) -> OAuthTokens:
    client_id = _env("ONEDRIVE_CLIENT_ID")
    client_secret = _env("ONEDRIVE_CLIENT_SECRET")
    resp = requests.post(
        _ONEDRIVE_TOKEN_URL,
        data={
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "scope": "https://graph.microsoft.com/Files.Read.All https://graph.microsoft.com/Sites.Read.All",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return OAuthTokens(
        access_token=data["access_token"],
        refresh_token=refresh_token,
        expires_in=int(data.get("expires_in", 3600)),
        provider="onedrive",
    )


# ── State token helpers ───────────────────────────────────────────────────────

def build_oauth_state(provider: str, return_url: str) -> str:
    """Pack provider + return_url into a single state token."""
    payload = json.dumps({"provider": provider, "return_url": return_url})
    return _base64url_encode(payload.encode())


def parse_oauth_state(state: str) -> tuple[str, str]:
    """Reverse build_oauth_state."""
    padding = 4 - (len(state) % 4)
    if padding < 4:
        state += "=" * padding
    payload = base64.urlsafe_b64decode(state).decode()
    data = json.loads(payload)
    return data["provider"], data["return_url"]
