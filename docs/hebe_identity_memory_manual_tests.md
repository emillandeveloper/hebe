# Hebe Identity And Memory Manual Tests

Run the backend, open the UI, and use `/debug/memory` to inspect persistent memory.

1. Feminine identity
   - User: `Hebe, estas listo?`
   - Expected: Hebe answers in feminine form, e.g. `Lista, Leo...`
   - Must not say: `Listo...`

2. Bilingual behavior
   - User: `hello hebe, are you there?`
   - Expected: English reply.
   - User: `hola hebe, estas ahi?`
   - Expected: Spanish from Spain.

3. No assistant tone
   - User: `quien eres?`
   - Expected: `Soy Hebe...` style answer.
   - Must not include: `soy una IA`, `asistente`, `en que puedo ayudarte`.

4. Memory write
   - User: `Recuerda que prefiero que hables en femenino y que si hablas espanol uses espanol de Espana.`
   - Expected: Hebe acknowledges naturally.
   - Check: `/debug/memory` includes a stable fact or chunk about feminine form and peninsular Spanish.

5. Memory retrieval after restart
   - Restart backend.
   - User: `Como deberias hablar?`
   - Expected: Hebe recalls feminine form and Spanish from Spain when speaking Spanish.

6. Twitch format
   - Twitch message: `hello hebe`
   - Expected: English, one line, under 240 chars.
   - Twitch message: `hola hebe`
   - Expected: Spanish from Spain, one line, under 240 chars.
