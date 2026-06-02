from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

import requests


class TwitchHelixClient:
    STREAMS_URL = "https://api.twitch.tv/helix/streams"
    CHANNELS_URL = "https://api.twitch.tv/helix/channels"

    def __init__(
        self,
        *,
        client_id: str,
        oauth_token: str,
        broadcaster_id: str,
        channel_name: str = "",
        session: Optional[requests.Session] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.client_id = str(client_id or "").strip()
        self.oauth_token = str(oauth_token or "").strip()
        self.broadcaster_id = str(broadcaster_id or "").strip()
        self.channel_name = str(channel_name or "").strip()
        self.timeout_sec = float(timeout_sec)
        self._session = session or requests.Session()
        self.last_status_by_endpoint: dict[str, int] = {}
        self.last_error_by_endpoint: dict[str, str] = {}

    def is_configured(self) -> bool:
        return bool(self.client_id and self.oauth_token and self.broadcaster_id)

    def get_stream(self) -> dict | None:
        data = self._get("get_streams", self.STREAMS_URL, {"user_id": self.broadcaster_id})
        rows = data.get("data") or []
        if not rows:
            return None
        return rows[0]

    def get_channel_info(self) -> dict | None:
        data = self._get("get_channel_information", self.CHANNELS_URL, {"broadcaster_id": self.broadcaster_id})
        rows = data.get("data") or []
        if not rows:
            return None
        return rows[0]

    def _get(self, endpoint_name: str, url: str, params: dict[str, str]) -> dict:
        missing = self._missing_config()
        path = urlparse(url).path or url
        token_had_oauth_prefix = self.oauth_token.lower().startswith("oauth:")
        print(
            "[HEBE][TWITCH][HELIX] request "
            f"endpoint={endpoint_name} path={path} "
            f"client_id_present={bool(self.client_id)} "
            f"oauth_token_present={bool(self.oauth_token)} "
            f"token_oauth_prefix_stripped={token_had_oauth_prefix}",
            flush=True,
        )
        if missing:
            raise RuntimeError(f"Missing Twitch config: {', '.join(missing)}")

        response = self._session.get(
            url,
            headers=self._build_headers(),
            params=params,
            timeout=self.timeout_sec,
        )
        print(
            f"[HEBE][TWITCH][HELIX] response endpoint={endpoint_name} status={response.status_code}",
            flush=True,
        )
        self.last_status_by_endpoint[endpoint_name] = int(response.status_code)
        if not response.ok:
            error_text = self._sanitize_error_body(response)
            self.last_error_by_endpoint[endpoint_name] = error_text
            print(
                f"[HEBE][TWITCH][HELIX] error endpoint={endpoint_name} "
                f"status={response.status_code} body={error_text}",
                flush=True,
            )
            raise RuntimeError(
                f"Helix {endpoint_name} failed: {response.status_code} "
                f"{self._status_label(response.status_code)} - {error_text}"
            )
        self.last_error_by_endpoint.pop(endpoint_name, None)
        return response.json()

    def _build_headers(self) -> dict[str, str]:
        token = self.oauth_token
        if token.lower().startswith("oauth:"):
            token = token.split(":", 1)[1]
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"

        return {
            "Authorization": token,
            "Client-Id": self.client_id,
        }

    def _missing_config(self) -> list[str]:
        missing = []
        if not self.client_id:
            missing.append("TWITCH_CLIENT_ID")
        if not self.oauth_token:
            missing.append("TWITCH_BROADCASTER_OAUTH_TOKEN or TWITCH_OAUTH_TOKEN")
        if not self.broadcaster_id:
            missing.append("TWITCH_BROADCASTER_ID")
        return missing

    def _sanitize_error_body(self, response) -> str:
        try:
            data = response.json()
            parts = [
                str(data.get("error") or "").strip(),
                str(data.get("message") or "").strip(),
            ]
            text = " - ".join(part for part in parts if part)
        except Exception:
            text = str(getattr(response, "text", "") or "").strip()
        if self.oauth_token and len(self.oauth_token) >= 8:
            text = text.replace(self.oauth_token, "[redacted]")
        return text[:500] or "no error body"

    def _status_label(self, status_code: int) -> str:
        labels = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            429: "Too Many Requests",
        }
        if status_code in labels:
            return labels[status_code]
        if 500 <= status_code:
            return "Twitch Server Error"
        return "HTTP Error"
