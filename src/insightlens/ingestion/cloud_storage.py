"""Cloud storage connectors: Google Drive, Dropbox, OneDrive.

Each connector:
  1. Returns a list of PDF files available in the connected account/folder.
  2. Downloads the raw bytes of a given file on demand.
  3. Does NOT store anything in Atticus — only retrieves content for ingestion.

Tokens are stored per-user in session_state and are NOT persisted server-side.
"""

from __future__ import annotations

import io
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import requests


@dataclass
class CloudFile:
    id: str          # provider-native file id
    name: str        # display name
    size_bytes: int
    mime_type: str   # "application/pdf" or "" if unknown
    web_url: str     # share URL (may require auth)


@dataclass
class CloudFolder:
    id: str
    name: str


class CloudProvider(ABC):
    """Base class for a cloud storage provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name shown in the UI."""
        raise NotImplementedError

    @abstractmethod
    def list_pdfs(self, folder_id: str | None = None) -> list[CloudFile]:
        """Return all PDF files in the given folder, or the root if None."""
        raise NotImplementedError

    @abstractmethod
    def download(self, file_id: str) -> tuple[bytes, str]:
        """Download file bytes and return (bytes, filename)."""
        raise NotImplementedError

    @abstractmethod
    def list_folders(self) -> list[CloudFolder]:
        """Return available top-level folders."""
        raise NotImplementedError


# ── Google Drive ────────────────────────────────────────────────────────────────

_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
]


def _gdrive_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}


class GoogleDriveProvider(CloudProvider):
    """Google Drive via the Drive API v3."""

    def __init__(self, access_token: str):
        self._token = access_token

    @property
    def provider_name(self) -> str:
        return "Google Drive"

    def list_folders(self) -> list[CloudFolder]:
        query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
        resp = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers=_gdrive_headers(self._token),
            params={"q": query, "fields": "files(id,name)", "pageSize": 100},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("files", [])
        return [CloudFolder(id=f["id"], name=f["name"]) for f in items]

    def list_pdfs(self, folder_id: str | None = None) -> list[CloudFile]:
        parent_clause = f"'{folder_id}' in parents" if folder_id else "'root' in parents"
        query = f"({parent_clause}) and mimeType='application/pdf' and trashed=false"
        resp = requests.get(
            "https://www.googleapis.com/auth/drive.files",
            headers=_gdrive_headers(self._token),
            params={
                "q": query,
                "fields": "files(id,name,size,mimeType,webViewLink)",
                "pageSize": 200,
                "orderBy": "name",
            },
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("files", [])
        return [
            CloudFile(
                id=f["id"],
                name=f.get("name", "untitled.pdf"),
                size_bytes=int(f.get("size") or 0),
                mime_type=f.get("mimeType", "application/pdf"),
                web_url=f.get("webViewLink", ""),
            )
            for f in items
        ]

    def download(self, file_id: str) -> tuple[bytes, str]:
        resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers=_gdrive_headers(self._token),
            params={"alt": "media"},
            timeout=60,
        )
        resp.raise_for_status()
        disposition = resp.headers.get("content-disposition", "")
        name = ""
        if "filename=" in disposition:
            name = disposition.split("filename=")[-1].strip('"')
        if not name:
            name = f"{file_id}.pdf"
        return resp.content, name


# ── Dropbox ────────────────────────────────────────────────────────────────────

_DROPBOX_API = "https://api.dropboxapi.com"
_DROPBOX_CONTENT = "https://content.dropboxapi.com"


class DropboxProvider(CloudProvider):
    """Dropbox via the Dropbox API v2."""

    def __init__(self, access_token: str):
        self._token = access_token

    @property
    def provider_name(self) -> str:
        return "Dropbox"

    def list_folders(self) -> list[CloudFolder]:
        resp = requests.post(
            f"{_DROPBOX_API}/2/files/list_folder",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            json={"path": "", "recursive": False},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("entries", [])
        return [
            CloudFolder(id=e["id"], name=e["name"])
            for e in items
            if e[".tag"] == "folder"
        ]

    def list_pdfs(self, folder_id: str | None = None) -> list[CloudFile]:
        path = folder_id or ""
        resp = requests.post(
            f"{_DROPBOX_API}/2/files/list_folder",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            json={"path": path, "include_media_info": True},
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("entries", [])
        return [
            CloudFile(
                id=e["id"],
                name=e["name"],
                size_bytes=e.get("size", 0),
                mime_type="application/pdf",
                web_url="",
            )
            for e in items
            if e[".tag"] == "file" and e["name"].lower().endswith(".pdf")
        ]

    def download(self, file_id: str) -> tuple[bytes, str]:
        resp = requests.post(
            f"{_DROPBOX_CONTENT}/2/files/download",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Dropbox-API-Arg": f'{{"path":"{file_id}"}}',
            },
            timeout=120,
        )
        resp.raise_for_status()
        name = resp.headers.get("dropbox-api-result", "{}")
        import json
        try:
            name = json.loads(name).get("name", file_id)
        except Exception:
            name = file_id
        return resp.content, name


# ── OneDrive / SharePoint ──────────────────────────────────────────────────────

_MS_SCOPES = [
    "https://graph.microsoft.com/Files.Read.All",
    "https://graph.microsoft.com/Sites.Read.All",
]


class OneDriveProvider(CloudProvider):
    """Microsoft OneDrive and SharePoint via MS Graph API."""

    def __init__(self, access_token: str):
        self._token = access_token

    @property
    def provider_name(self) -> str:
        return "OneDrive"

    def list_folders(self) -> list[CloudFolder]:
        resp = requests.get(
            "https://graph.microsoft.com/v1.0/me/drive/root/children",
            headers={"Authorization": f"Bearer {self._token}"},
            params={"$filter": "folder ne null", "$select": "id,name"},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("value", [])
        return [
            CloudFolder(id=c["id"], name=c["name"])
            for c in items
            if c.get("folder")
        ]

    def list_pdfs(self, folder_id: str | None = None) -> list[CloudFile]:
        if folder_id:
            url = f"https://graph.microsoft.com/v1.0/me/drive/items/{folder_id}/children"
        else:
            url = "https://graph.microsoft.com/v1.0/me/drive/root/children"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._token}"},
            params={"$filter": "file/mimeType eq 'application/pdf'", "$select": "id,name,size,webUrl"},
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("value", [])
        return [
            CloudFile(
                id=c["id"],
                name=c["name"],
                size_bytes=int(c.get("size") or 0),
                mime_type="application/pdf",
                web_url=c.get("webUrl", ""),
            )
            for c in items
            if c.get("file", {}).get("mimeType") == "application/pdf"
        ]

    def download(self, file_id: str) -> tuple[bytes, str]:
        resp = requests.get(
            f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content",
            headers={"Authorization": f"Bearer {self._token}"},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        name = file_id
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            name = cd.split("filename=")[-1].strip('"')
        return resp.content, name


# ── Factory ────────────────────────────────────────────────────────────────────

def make_provider(
    provider: str,
    access_token: str,
) -> CloudProvider:
    """Return the correct provider instance for the given provider key."""
    if provider == "google_drive":
        return GoogleDriveProvider(access_token)
    elif provider == "dropbox":
        return DropboxProvider(access_token)
    elif provider == "onedrive":
        return OneDriveProvider(access_token)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Convenience: download to a local temp Path ─────────────────────────────────

def download_to_temp(provider: CloudProvider, file_id: str) -> Path:
    """Download a cloud file into a temp directory and return the Path."""
    bytes_, name = provider.download(file_id)
    tmp = Path(tempfile.gettempdir()) / f"atticus_cloud_{file_id}_{name}"
    tmp.write_bytes(bytes_)
    return tmp