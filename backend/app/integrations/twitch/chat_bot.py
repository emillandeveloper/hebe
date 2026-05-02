from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Optional

from websockets.sync.client import connect


class TwitchChatBot:
    WS_URI = "wss://irc-ws.chat.twitch.tv:443"
    PING_RESPONSE = "PONG :tmi.twitch.tv"

    def __init__(
        self,
        *,
        channel_name: str,
        bot_username: str,
        oauth_token: str,
        enabled: bool = True,
        message_callback: Optional[Callable[[str, str, str, str], None]] = None,
        reconnect_delay: float = 5.0,
    ) -> None:
        self.channel_name = str(channel_name or "").strip().lower()
        self.bot_username = str(bot_username or "").strip()
        self.oauth_token = str(oauth_token or "").strip()
        if self.oauth_token.lower().startswith("oauth:"):
            self.oauth_token = self.oauth_token.split(":", 1)[1]
        self.enabled = enabled
        self.message_callback = message_callback
        self.reconnect_delay = reconnect_delay

        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._connected = False

        # Continuidad conversacional corta:
        # si Hebe acaba de preguntar algo a un chatter, el siguiente mensaje
        # de ese chatter puede entrar aunque no vuelva a mencionar "hebe".
        self.followup_enabled = os.getenv("HEBE_TWITCH_FOLLOWUP_ENABLED", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.pending_reply_seconds = float(os.getenv("HEBE_TWITCH_PENDING_REPLY_SECONDS", "90"))
        self._pending_reply_until: dict[str, float] = {}
        self._last_callback_username: str | None = None
        self._last_callback_at: float = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        if not self.enabled:
            print("[HEBE][TWITCH][CHATBOT] disabled", flush=True)
            return False

        if self._thread and self._thread.is_alive():
            return True

        self._stop = False
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop = True
        self._connected = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _run_forever(self) -> None:
        first_attempt = True
        while not self._stop:
            try:
                self._run_once()
            except Exception as exc:
                print(f"[HEBE][TWITCH][CHATBOT] error: {exc!r}", flush=True)
            if self._stop:
                break
            if not first_attempt:
                print(
                    f"[HEBE][TWITCH][CHATBOT] reconnecting in {self.reconnect_delay} seconds...",
                    flush=True,
                )
            first_attempt = False
            time.sleep(self.reconnect_delay)

    def _run_once(self) -> None:
        if not self._connect():
            return

        while not self._stop and self._ws is not None:
            try:
                line = self._ws.recv()
            except Exception as exc:
                print(f"[HEBE][TWITCH][CHATBOT] recv failed: {exc!r}", flush=True)
                break

            if line is None:
                break

            line = str(line).strip()
            if not line:
                continue

            if line.startswith("PING"):
                print("[HEBE][TWITCH][CHATBOT] IRC PING received", flush=True)
                self._send_raw(self.PING_RESPONSE)
                continue

            if "PRIVMSG" in line:
                print(f"[HEBE][TWITCH][CHATBOT] raw IRC PRIVMSG line: {line!r}", flush=True)
                self._handle_privmsg(line)
            else:
                print(
                    f"[HEBE][TWITCH][CHATBOT] raw IRC server line (non-chat): {line!r}",
                    flush=True,
                )

        self._cleanup_connection()

    def _connect(self) -> bool:
        if not self.enabled:
            return False

        missing = []
        if not self.channel_name:
            missing.append("channel_name")
        if not self.bot_username:
            missing.append("bot_username")
        if not self.oauth_token:
            missing.append("oauth_token")

        if missing:
            print(
                f"[HEBE][TWITCH][CHATBOT] missing config: {', '.join(missing)}",
                flush=True,
            )
            return False

        try:
            self._ws = connect(self.WS_URI, open_timeout=10, close_timeout=10)
        except Exception as exc:
            print(f"[HEBE][TWITCH][CHATBOT] connect failed: {exc!r}", flush=True)
            self._cleanup_connection()
            return False

        try:
            self._send_raw(f"PASS oauth:{self.oauth_token}")
            self._send_raw(f"NICK {self.bot_username}")
            self._send_raw("CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership")
            self._send_raw(f"JOIN #{self.channel_name}")
            self._connected = True
            print(
                f"[HEBE][TWITCH][CHATBOT] connected bot={self.bot_username!r} channel=#{self.channel_name}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[HEBE][TWITCH][CHATBOT] login failed: {exc!r}", flush=True)
            self._cleanup_connection()
            return False

    def _cleanup_connection(self) -> None:
        if self._connected:
            print("[HEBE][TWITCH][CHATBOT] disconnected", flush=True)
        self._connected = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _send_raw(self, data: str) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected")

        # Nunca imprimir tokens reales en logs. Twitch usa PASS oauth:<token>.
        log_data = data
        if data.upper().startswith("PASS "):
            log_data = "PASS oauth:****"

        print(f"[HEBE][TWITCH][CHATBOT] sending raw: {log_data!r}", flush=True)
        self._ws.send(data + "\r\n")

    def _parse_privmsg_line(self, line: str) -> tuple[str, str, str] | None:
        """Parse a PRIVMSG line, handling optional IRCv3 tags from Twitch."""
        if line.startswith("@"):
            parts = line.split(" ", 1)
            if len(parts) != 2:
                return None
            line = parts[1]

        if line.startswith(":"):
            line = line[1:]

        parts = line.split(" ", 3)
        if len(parts) != 4:
            return None

        prefix, command, channel, message = parts
        if command != "PRIVMSG":
            return None

        return prefix, channel, message

    def _handle_privmsg(self, line: str) -> None:
        parsed = self._parse_privmsg_line(line)
        if parsed is None:
            print(
                f"[HEBE][TWITCH][CHATBOT] failed to parse PRIVMSG line: {line!r}",
                flush=True,
            )
            return

        prefix, channel, message = parsed

        if message.startswith(":"):
            message = message[1:]

        if prefix.startswith(":"):
            prefix = prefix[1:]

        username = prefix.split("!", 1)[0] if prefix else ""

        if not username:
            return

        # Twitch nos devuelve también los mensajes enviados por el bot.
        # Los usamos para saber si Hebe acaba de abrir una pregunta/turno.
        if username.lower() == self.bot_username.lower():
            self._handle_own_bot_message(message)
            return

        self._cleanup_pending_replies()

        has_mention = bool(re.search(r"\b(?:hebe|ebe)\b", message, flags=re.IGNORECASE))

        # Importante: si el mensaje ya menciona a Hebe, NO consumimos el pending_reply.
        # Caso real:
        #   Hebe: "¿qué tal va la prueba?"  -> abre pending
        #   Leo: "pues he tenido un día del revés hebe XD" -> entra por mención
        #   Leo: "fue un problema personal..." -> debe entrar como follow-up sin mención
        # Si consumimos el pending en el segundo mensaje, el tercero se ignora.
        has_pending_reply = False if has_mention else self._consume_pending_reply(username)

        if has_mention and self._has_pending_reply(username):
            print(
                f"[HEBE][TWITCH][CHATBOT] message has mention; keeping pending reply open "
                f"user={username!r} message={message!r}",
                flush=True,
            )

        if not has_mention and not has_pending_reply:
            print(
                f"[HEBE][TWITCH][CHATBOT] ignored chat message without mention: {message!r}",
                flush=True,
            )
            return

        if has_pending_reply and not has_mention:
            print(
                f"[HEBE][TWITCH][CHATBOT] accepted follow-up without mention user={username!r} message={message!r}",
                flush=True,
            )

        if self.message_callback is not None:
            print(
                f"[HEBE][TWITCH][CHATBOT] incoming message user={username!r} channel={channel!r} message={message!r}",
                flush=True,
            )
            self._last_callback_username = username
            self._last_callback_at = time.time()
            self.message_callback(
                username,
                username,
                message,
                channel,
            )

    def _handle_own_bot_message(self, message: str) -> None:
        if not self.followup_enabled:
            return

        if not self._last_callback_username:
            return

        # Solo asociamos el eco del bot con la última interacción si acaba
        # de ocurrir. Evita abrir pending_reply por mensajes antiguos o ecos raros.
        if time.time() - self._last_callback_at > 15:
            return

        if not self._opens_followup(message):
            return

        username_key = self._last_callback_username.lower()
        expires_at = time.time() + max(1.0, self.pending_reply_seconds)
        self._pending_reply_until[username_key] = expires_at

        print(
            f"[HEBE][TWITCH][CHATBOT] opened pending reply user={self._last_callback_username!r} "
            f"seconds={self.pending_reply_seconds:g} bot_message={message!r}",
            flush=True,
        )

    def _opens_followup(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False

        # Preguntas explícitas o frases que invitan al chatter a contestar.
        if "?" in text or "¿" in text:
            return True

        followup_markers = (
            "dime",
            "cuéntame",
            "cuentame",
            "dispara",
            "te leo",
            "qué pasó",
            "que pasó",
            "qué cuentas",
            "que cuentas",
            "qué te ha",
            "que te ha",
            "me cuentas",
            "a ver qué",
            "a ver que",
        )

        return any(marker in text for marker in followup_markers)

    def _cleanup_pending_replies(self) -> None:
        if not self._pending_reply_until:
            return

        now = time.time()
        expired = [key for key, until in self._pending_reply_until.items() if until <= now]
        for key in expired:
            self._pending_reply_until.pop(key, None)

    def _has_pending_reply(self, username: str) -> bool:
        if not self.followup_enabled:
            return False

        key = (username or "").lower().strip()
        if not key:
            return False

        until = self._pending_reply_until.get(key)
        if until is None:
            return False

        if until <= time.time():
            self._pending_reply_until.pop(key, None)
            return False

        return True

    def _consume_pending_reply(self, username: str) -> bool:
        if not self._has_pending_reply(username):
            return False

        key = (username or "").lower().strip()
        self._pending_reply_until.pop(key, None)
        return True
