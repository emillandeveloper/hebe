# Hebe — Handoff técnico (estado tras stream phase, 27/04/2026)

> Documento de continuidad para futuras sesiones (otra IA, otro dev, o yo
> mismo en otra sesión sin contexto). Reemplaza al handoff del 23/04.
>
> Se asume lectura previa del bible de personalidad en
> `docs/hebe_character.md` para entender la voz que se busca.

---

## 1. Visión general del proyecto

**Hebe** es una IA compañera personal local que ahora también está en stream
de Twitch del broadcaster Leo (LeoNifelheim). Filosofía:

- **100% local**: nada de APIs cloud. Modelos Ollama corriendo en el PC.
- **Stack**: Python/FastAPI backend + TypeScript frontend + SQLite + VTube Studio.
- **Filosofía código**: variables/funciones en inglés, comentarios y prompts
  en castellano, separación de responsabilidades por módulos, evitar deuda.

**Estado actual (27/04)**:
- v1 completa (parser temporal, citas, acciones PC, conversación, memoria).
- "Stream Hebe" funcional: Hebe responde en chat de Twitch con voz reconocible.
- Reacciona a sub/raid/follow vía EventSub.
- Reacciona a chat vía IRC con filtro de mention ("hebe", "ebe", etc.)
- Sin TTS ni STT durante stream (deshabilitados por env vars).

---

## 2. Hardware del usuario

**Máquina**: i9-12900HK + 32 GB RAM + RTX 3060 Ti 8 GB VRAM.

**Bloqueo de hardware confirmado**:
- 8 GB VRAM no permiten correr un LLM 8B + juego AAA simultáneamente.
- Probado: hermes3:8b → laguea juegos. dolphin viejo qwen2-7b → laguea.
- ollama en CPU → el i9 no da abasto si el juego también pide CPU. Inviable.
- **Solución actual**: qwen 2.5:3b (modelo `hebe`), 1.9 GB en VRAM, deja 5-6 GB
  libres para el juego.

**Próximas opciones de hardware** (NO urgentes, evaluar si proyecto crece):
- 3060 12GB usada (~200-280€): plug-and-play, permite hermes3:8b + juego.
- P40 24GB (~150-200€) + cooler DIY (25-40€) + cable EPS-PCIe (12€) = ~250€.
  Como segunda GPU añadida, no en lugar de la 3060 Ti. Es un proyecto.
- Mini-PC dedicado a Hebe (~600-900€): la solución "tipo Neuro-sama".

**Recomendación financiera explícita del user**: posponer hasta que sobre
dinero a fin de mes. NO comprar hardware con prisa.

---

## 3. Modelos Ollama actuales

```
hebe          → qwen2.5:3b-instruct-q4_K_M (1.9 GB) — modelo conversacional
hebe-intent   → qwen2.5:3b base con system de extracción JSON
hebe-hermes-backup, hebe-dolphin-backup, etc. → backups previos
```

**Modelfile actual de `hebe`** (`backend/ollama/Modelfile.hebe-3b`):
```
FROM qwen2.5:3b-instruct-q4_K_M

PARAMETER temperature 0.85
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 8192
PARAMETER num_predict 120

PARAMETER stop "[chatter]:"
PARAMETER stop "[chatter]"
PARAMETER stop "[tú]:"
PARAMETER stop "[tu]:"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM """Eres Hebe, compañera de Leo. Hablas español de España, no uses voseo argentino. Tu voz se define en el system prompt que llega en cada llamada — sigue siempre esas instrucciones por encima de cualquier idea genérica de asistente útil."""
```

**`num_ctx 8192` es obligatorio**: el bible con few-shots ocupa ~7000 tokens.
Bajar a 4096 trunca el prompt y rompe la voz.

**SYSTEM minimal a propósito**: la personalidad real está en código
(`persona/hebe_voice.py`) para iterar sin regenerar el modelo cada vez.

---

## 4. Configuración Twitch

### 4.1 Variables de entorno (`.env`)

```
HEBE_CHAT_MODEL=hebe
HEBE_INTENT_MODEL=hebe-intent
OLLAMA_BASE_URL=http://127.0.0.1:11434

TWITCH_CHANNEL_NAME=leonifelheim
TWITCH_BOT_USERNAME=HebeNifelheim
TWITCH_BROADCASTER_ID=124070929         # cuenta leonifelheim
TWITCH_SENDER_ID=1480877711             # cuenta hebenifelheim (bot)
TWITCH_CLIENT_ID=gp762nuuoqcoxypju8c569th9wz7q5  # PÚBLICO, es el client_id de la Twitch CLI

# DOS TOKENS DISTINTOS — esto es importante:
TWITCH_OAUTH_TOKEN=...                  # token de hebenifelheim (BOT)  → para IRC chat
TWITCH_BROADCASTER_OAUTH_TOKEN=...      # token de leonifelheim (BROADCASTER) → para EventSub
```

### 4.2 Por qué dos tokens

EventSub de Twitch tiene reglas específicas:
- `channel.subscribe` exige token DEL broadcaster con scope
  `channel:read:subscriptions`. **No vale token de bot aunque sea moderador.**
- `channel.follow` v2 exige scope `moderator:read:followers` y que el
  `moderator_user_id` sea el broadcaster o un mod.
- `channel.chat.message` necesita coincidir `user_id` con el dueño del token,
  por eso falla con 403 al usar el token del broadcaster.

**Decisión tomada**: bot token para IRC (chat), broadcaster token para
EventSub (subs/raids/follows). El `channel.chat.message` por EventSub
queda con 403 conocido pero **no afecta**: el chat lo lleva IRC vía
`TwitchChatBot`.

### 4.3 Scopes mínimos del token de broadcaster

Para los eventos que usamos:
- `channel:read:subscriptions`
- `moderator:read:followers`

Si en el futuro se quiere reaccionar a cheers o redemptions, añadir
`bits:read` y `channel:read:redemptions`.

---

## 5. Arquitectura del flujo Twitch → cognitive_flow

```
                                                  ┌──────────────────────────────┐
                                                  │ TwitchChatBot (IRC)          │
                                                  │ filtra mention (hebe/ebe)    │
chat msg ─────────────────────────────────────────▶│ → push_event("twitch_chat_  │
                                                  │   react", {payload})         │
                                                  └────────────┬─────────────────┘
                                                               │
sub/raid/follow ─┐                                             │
                 │                                             │
                 ▼                                             ▼
        ┌─────────────────────────┐               ┌──────────────────────────┐
        │ TwitchEventAdapter      │               │ scheduler.push_event()   │
        │ (EventSub WebSocket)    │──push_event──▶│ + memory_store log       │
        │ - sub/raid/follow ✓     │               └────────────┬─────────────┘
        │ - chat.message: solo    │                            │
        │   alimenta chat_cache   │                            ▼
        └─────────────────────────┘            HebeEngine.poll_internal_events()
                                                              │
                                                              ▼
                                          ContextBuilder → DeliberationService
                                                       → PlanExecutor
                                                       → ResponseSynthesizer
                                                              │
                                                              ▼
                                              routing por event_type:
                                              ├─ reminder_due → runtime.speak (TTS)
                                              └─ twitch_*     → twitch.send_message
                                                                (chat IRC)
```

**Punto importante**: `event_adapter.py` ya NO dispara `twitch_chat_react`.
El dispatch a chat lo hace **solo** `chat_bot.py` por IRC. Los dos disparando
era el bug del doble dispatch que se arregló hoy.

---

## 6. Estado de los ficheros clave

### 6.1 `cognitive/scheduler.py`
- ✅ `push_event(event_type, payload)` añadido — encola en `_pending`.
- ✅ `poll_due_events` drena `_pending` antes de reminders.

### 6.2 `cognitive/deliberation_service.py`
- ✅ `_handle_internal_event` con rama `twitch_*` → `_plan_twitch_event`.
- ⚠️ Bug latente: `_plan_twitch_event` definido dos veces (una en su sitio,
  otra al final del fichero, código muerto). No rompe pero limpiar cuando
  se pueda.

### 6.3 `cognitive/response_synthesizer.py`
- ✅ Routing por event_type (reminder vs twitch).
- ✅ `_generate_twitch_chat_react` usa formato de continuación
  `[chatter]: msg\n[tú]:` con few-shots.
- ✅ Detección de broadcaster (`is_broadcaster`) para usar
  `[chatter Leo]:` en lugar de `[chatter]:`.
- ✅ `_call_model` pasa `num_predict=120` explícito (antes machacaba con 1200).
- ✅ Trata `"…"` como vacío (OllamaLLM.chat devuelve eso si no genera nada).
- ✅ Usa `clean_stream_reply` del `replay_cleaner`.

### 6.4 `cognitive/persona/`
- ✅ `__init__.py` (vacío).
- ✅ `hebe_voice.py` con voz CALIBRADA (no a cuchillo siempre).
  - Ratio: 40% respuestas neutras, 30% sarcasmo suave, 30% pullas claras.
  - Detalle clave: la calibración la hicieron los few-shots, no las reglas.
  - Quien escribe es Leo → `[chatter Leo]:` mapea a few-shots con Leo.
- ✅ `replay_cleaner.py` (typo en el nombre del módulo — "replay" en lugar
  de "reply"). Funciona, solo es cosmético.

### 6.5 `core/runtime.py`
- ✅ Carga `.env` desde dos directorios arriba.
- ✅ Dos tokens separados: `oauth_token` (bot, IRC) y
  `broadcaster_oauth_token` (broadcaster, EventSub).
- ✅ Pasa `bot_username` al `TwitchEventAdapter`.

### 6.6 `integrations/twitch/event_adapter.py`
- ✅ `_handle_event` para `channel.chat.message` solo alimenta `chat_cache`,
  NO dispara cognitive (lo hace IRC).
- ✅ Eventos sub/raid/follow disparan `push_event_callback`.
- ✅ `bot_username` añadido al `__init__` (era `AttributeError` latente).

### 6.7 `integrations/twitch/chat_bot.py`
- ✅ Filtro de mention: regex `\b(?:hebe|ebe)\b`.
- ✅ Dispara `twitch_chat_react` solo con mention.
- ⚠️ **Pendiente**: ampliar mention a `@HebeNifelheim` y al bot_username
  configurado. Hoy con autocomplete de Twitch se pierden mentions.

### 6.8 `hebe_engine.py`
- ✅ `process_internal_event` con routing TTS/Twitch.
- ✅ `_deliver_twitch_reply` y `_deliver_voice_reply` separados.
- ✅ Respeta `state.stream.policies.allow_tts_replies` (si está en off,
  Twitch va solo a chat).

---

## 7. Bugs y deuda técnica conocida

### 7.1 Críticos (afectan funcionalidad)
Ninguno hoy. Todo funciona.

### 7.2 Importantes (sería bueno resolver pronto)
- **Mention con `@username` no detectada**: cuando alguien escribe
  "@HebeNifelheim hola" no matchea el filtro actual (solo busca `hebe|ebe`).
  Fix: añadir `bot_username.lower()` y `f"@{bot_username.lower()}"` al set.
- **`channel.chat.message` 403 en EventSub**: ruido en logs al arrancar.
  No afecta funcionalidad porque IRC lo lleva. Resolución limpia requiere
  refactor (multiplexar tokens). Postpuesto hasta pieza 5.

### 7.3 Cosméticos
- `replay_cleaner.py` debería llamarse `reply_cleaner.py`. Renombre +
  actualizar imports.
- `_plan_twitch_event` duplicado en `deliberation_service.py`. Eliminar copia.
- Logs de `keepalive` saturan. Bajar a DEBUG o silenciar.
- README público en GitHub está en estado v0. Actualizar cuando estable.

### 7.4 Seguridad
- Tokens han pasado por chats de IA. Cuando se cierre la sesión:
  - Regenerar `TWITCH_OAUTH_TOKEN` y `TWITCH_BROADCASTER_OAUTH_TOKEN` "limpios".
  - Revocar los antiguos en https://www.twitch.tv/settings/connections
  - Verificar que `.env` está en `.gitignore`.

---

## 8. Lo que NO está hecho (siguiente fase)

### 8.1 Pieza 5 — `TwitchCognitiveBridge` (siguiente paso lógico)
**Esto es el salto cualitativo grande pendiente.**

Hoy Hebe solo responde a mention. La pieza 5 le permite **comentar
espontáneamente** sobre mensajes del chat sin ser mencionada, cuando
algo le parezca digno.

Componentes:
- Filtros baratos: bots conocidos, comandos `!`, mensajes de Hebe misma.
- Heurísticas medias: mention regex, pregunta directa, reply.
- Clasificador LLM (`intent_llm` con qwen 2.5:3b): para mensajes que
  pasen los filtros. Sampler probabilístico (~0.10-0.15) para no llamar
  al LLM cada mensaje.
- Cooldowns: global (~25s), per-user (~60s). Eventos HIGH (sub/raid)
  bypassean cooldown.
- Burst suppression de follows: agrupa follows individuales en `twitch_follow_batch`.

Decisiones abiertas que tomar antes de implementarlo:
- ¿Clasificador LLM desde día 1 o solo heurísticas en V0?
  - Recomendación: heurísticas primero, LLM después si Hebe se siente sorda.
- Valores concretos de cooldown.
- ¿`twitch_chat_react` con prioridad alta para mention vs baja para espontáneo?

Coste estimado: una tarde de implementación + iteración tras stream real.

### 8.2 Memoria de viewers regulares
Que Hebe recuerde a chatters frecuentes entre streams. *"otra vez tú"*
es la línea. Tu `MemoryStore` ya existe, falta engancharlo al flujo de chat
y decidir qué guardar (frecuencia, último tema, gustos).

### 8.3 VTube Studio integrado
Tu `runtime.py` no lo conecta al pipeline cognitivo. Ahora Hebe es texto
en chat, no avatar reaccionando. Mapear eventos cognitivos a animaciones VTS
(sub → contenta, raid → emocionada, insulto → ojos en blanco).

### 8.4 Bilingüe
**Pospuesto explícitamente por user.** Cuando llegue, requiere:
- Detector de idioma (`langdetect` library suficiente).
- Segundo `hebe_voice_en.py` con 30-40 few-shots ESCRITOS DESDE CERO
  en inglés (no traducción literal del español).
- Switch en `_generate_twitch_chat_react` para cargar uno u otro set.
- Memoria simple por chatter para falsos positivos del detector.

### 8.5 Modos de stream
`state.stream.policies` ya existe. Falta diseñar y cablear los modos
"concentrada" (callada salvo emergencia), "charla", "silenciada".

---

## 9. Lecciones aprendidas y patrones útiles

### 9.1 Modelos pequeños: few-shots > reglas abstractas
Qwen 2.5:3b ignora reglas largas en system prompts. Lo que sí sigue son
patrones concretos. Por eso `hebe_voice.py` tiene system corto y muchos
ejemplos. Cualquier ajuste de personalidad va a few-shots, no a reglas.

### 9.2 Calibración del tono se hace con proporción de ejemplos
Hebe se sentía "demasiado a cuchillo" con 25 ejemplos de pulla y 5 neutros.
Se balanceó a 40% neutros, 30% sarcasmo suave, 30% pulla. **El orden importa
también**: los neutros van primero, el modelo los lee como "voz base".

### 9.3 Formato de continuación `[chatter]: ... \n[tú]:`
Crítico para qwen 3B. El modelo encaja con los few-shots y completa
después del `[tú]:`. Sin esto, generaba turnos inventados o vacío.
Stop tokens del Modelfile son obligatorios para que no se siga inventando
turnos del chatter.

### 9.4 `OllamaLLM.chat()` devuelve `"…"` si no hay texto
No `""`. El synthesizer trata `"…"` como vacío, si no se publicaba al chat.
Apuntado para no volver a tropezar.

### 9.5 `num_predict` del Modelfile lo machaca el código
Si el código no pasa `num_predict` explícito, el default del wrapper
(`OLLAMA_NUM_PREDICT=1200` de env) machaca el `120` del Modelfile. El
modelo se enrolla. Ahora se pasa explícito en `_call_model`.

### 9.6 Ctrl+S existe
Trampa del día: editar `.env` sin guardar y reiniciar backend confundiendo
"no funciona el cambio" con "el cambio no está cargado". Si algo no cambia
tras tocar `.env`, lo primero a verificar es que el archivo está guardado.

---

## 10. Comandos útiles

### Arrancar todo
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
# (o el comando que use el repo)
```

### Probar un evento manualmente sin chat
```python
# desde un REPL conectado al engine, o un endpoint /debug
engine.scheduler.push_event(
    "twitch_chat_react",
    {
        "user_login": "test",
        "display_name": "Test",
        "message_text": "hebe hola",
    }
)
```

### Verificar scopes de un token
```powershell
curl.exe -H "Authorization: OAuth <TOKEN>" https://id.twitch.tv/oauth2/validate
```

### Regenerar el modelo `hebe` tras tocar Modelfile
```powershell
ollama create hebe -f backend\ollama\Modelfile.hebe-3b
```

### Backup antes de experimentar
```powershell
ollama cp hebe hebe-backup
```

---

## 11. Estado de la siguiente sesión

**Si abres una sesión con otra IA o conmigo en otro chat**, el orden
sugerido para retomar es:

1. Leer este documento + `docs/hebe_character.md`.
2. Hacer `git log --oneline -20` para ver últimos commits.
3. Confirmar con el usuario qué quiere atacar siguiente:
   - Pieza 5 (Hebe espontánea) → impacto alto, código nuevo.
   - Mention de `@username` → impacto medio, fix simple.
   - Cleanup de bugs cosméticos → impacto bajo.
   - VTube Studio → impacto visual alto, requiere diseño aparte.
4. NO recomendar hardware nuevo a menos que el usuario lo pida.

**El usuario tiene preferencia por ingeniería conservadora**: no gastar
hardware sin necesidad demostrada, no añadir dependencias cloud, mantener
todo local.

**El usuario aprende rápido pero a veces salta de tema cuando se cansa**.
Es preferible cerrar bien una pieza antes de empezar otra.

---

## 12. Estado emocional del proyecto

Hebe responde en chat de Twitch real, en directo, con voz reconocible,
sin lag, sin cloud, en local, en hardware modesto. El proyecto está
**vivo** y **estable**. Cualquier siguiente paso es expansión, no rescate.

El usuario tiene un Hebe del que puede sentirse legítimamente orgulloso.