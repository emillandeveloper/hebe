# backend/app/services/vts_client.py
import os
import json
import asyncio
import websockets

VTS_HOST = os.getenv("HEBE_VTS_HOST", "127.0.0.1")
VTS_PORT = int(os.getenv("HEBE_VTS_PORT", "8001"))

VTS_PLUGIN_NAME = os.getenv("HEBE_VTS_PLUGIN_NAME", "HebeAssistant")
VTS_PLUGIN_AUTHOR = os.getenv("HEBE_VTS_PLUGIN_AUTHOR", "Leo")
VTS_PLUGIN_ICON = None

VTS_TOKEN_FILE = os.getenv("HEBE_VTS_TOKEN_FILE", "vts_auth_token.txt")


class VTSClient:
    def __init__(self, host=VTS_HOST, port=VTS_PORT):
        self.host = host
        self.port = port
        self.ws = None
        self.authenticated = False
        self.auth_token = None

        if os.path.exists(VTS_TOKEN_FILE):
            try:
                with open(VTS_TOKEN_FILE, "r", encoding="utf-8") as f:
                    self.auth_token = f.read().strip() or None
            except Exception:
                self.auth_token = None

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        print(f"🔌 Conectando a VTube Studio en {uri}...")
        self.ws = await websockets.connect(uri)
        await self.authenticate()

    async def request_auth_token(self):
        if self.ws is None:
            raise RuntimeError("WebSocket no inicializado")

        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "token-1",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": VTS_PLUGIN_NAME,
                "pluginDeveloper": VTS_PLUGIN_AUTHOR,
                "pluginIcon": VTS_PLUGIN_ICON,
            },
        }
        await self.ws.send(json.dumps(msg))
        print("📨 Enviado AuthenticationTokenRequest (acepta el plugin en VTS)")

        resp_raw = await self.ws.recv()
        resp = json.loads(resp_raw)

        if resp.get("messageType") == "AuthenticationTokenResponse":
            token = resp.get("data", {}).get("authenticationToken")
            if not token:
                raise RuntimeError("VTS no devolvió token de autenticación.")

            self.auth_token = token
            with open(VTS_TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(token)
            print("✅ Token de VTS guardado en", VTS_TOKEN_FILE)
        else:
            raise RuntimeError(f"Respuesta inesperada al pedir token: {resp}")

    async def authenticate(self):
        if self.ws is None:
            raise RuntimeError("WebSocket no inicializado")

        if self.auth_token is None:
            await self.request_auth_token()

        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-1",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": VTS_PLUGIN_NAME,
                "pluginDeveloper": VTS_PLUGIN_AUTHOR,
                "pluginIcon": VTS_PLUGIN_ICON,
                "authenticationToken": self.auth_token,
            },
        }
        await self.ws.send(json.dumps(msg))
        resp_raw = await self.ws.recv()
        resp = json.loads(resp_raw)

        if resp.get("messageType") == "AuthenticationResponse" and resp.get("data", {}).get("authenticated"):
            self.authenticated = True
            print("✅ Autenticado con VTube Studio.")
        else:
            raise RuntimeError(f"No se pudo autenticar con VTS: {resp}")

    async def _send_request(self, message_type: str, data: dict, request_id: str = "req-1"):
        if self.ws is None:
            raise RuntimeError("WebSocket no inicializado")

        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": message_type,
            "data": data,
        }
        await self.ws.send(json.dumps(msg))
        resp_raw = await self.ws.recv()
        return json.loads(resp_raw)

    async def list_hotkeys(self):
        return await self._send_request("HotkeysInCurrentModelRequest", {}, request_id="hotkeys-current-model-1")

    async def get_hotkey_id_by_name(self, hotkey_name: str):
        resp = await self._send_request("HotkeysInCurrentModelRequest", {}, request_id="hotkeys-lookup")
        data = resp.get("data", {}) or {}
        hotkeys = data.get("availableHotkeys") or data.get("hotkeys") or []
        for hk in hotkeys:
            if hk.get("name") == hotkey_name:
                return hk.get("hotkeyID")
        return None

    async def trigger_hotkey(self, hotkey_name: str):
        hotkey_id = await self.get_hotkey_id_by_name(hotkey_name)
        if not hotkey_id:
            print(f"❌ No se pudo obtener hotkeyID para '{hotkey_name}'.")
            return None
        return await self._send_request(
            "HotkeyTriggerRequest",
            {"hotkeyID": hotkey_id},
            request_id=f"hotkey-{hotkey_name}",
        )

    async def close(self):
        if self.ws:
            await self.ws.close()
        self.ws = None
        self.authenticated = False
        print("🔌 Desconectado de VTS.")


async def _vts_hotkey_async(nombre_hotkey: str):
    client = VTSClient()
    try:
        await client.connect()
        await client.trigger_hotkey(nombre_hotkey)
    finally:
        await client.close()


def vts_hotkey(nombre_hotkey: str):
    try:
        asyncio.run(_vts_hotkey_async(nombre_hotkey))
    except Exception as e:
        print(f"❌ Error al disparar hotkey VTS '{nombre_hotkey}': {e}")
