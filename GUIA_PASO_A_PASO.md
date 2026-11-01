# Guía paso a paso (según la guía del profesor)

Todo corre **en CPU** en la laptop que presenta. El token nunca se escribe en archivos ni se sube a Git. Flujo:

```
laptop: API (puerto 8000)  ->  túnel (URL pública https)  ->  pruebas curl desde otra red
```

## 1. Arrancar la API

Necesitas el proyecto, el modelo `modelo-final-v3.zip` (los pesos no van en Git) y un token (en el dry run sirve uno inventado; el real lo da Alfonso el día de la presentación).

1. Extrae el modelo para que los archivos queden **directo** en `weights/modelo-final-v3/` (`model.safetensors`, `config.json`, `metrics.json`, archivos del tokenizer), sin carpeta repetida.
2. Define el token y el nombre del equipo **en esa terminal**.

**Windows (PowerShell):**
```powershell
$env:QA_API_TOKEN = "<token>"
$env:TEAM_NAME = "<nombre-del-equipo>"
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

**Linux / macOS / WSL:**
```bash
export QA_API_TOKEN="<token>"
export TEAM_NAME="<nombre-del-equipo>"
./run.sh
```

La primera vez crea `.venv` e instala dependencias. Deja esa terminal abierta. Comprueba en otra:

```bash
curl http://localhost:8000/health        # Windows: curl.exe
```
Debe responder `200 {"status": "ok"}` (`503 {"status": "loading"}` mientras carga el modelo).

## 2. Túnel público

Usa **una** herramienta y deja instalada y probada una **segunda** como respaldo (ngrok, Cloudflare Quick Tunnel o Tailscale Funnel). Con la API corriendo en el puerto 8000:

```bash
# ngrok (una vez: ngrok config add-authtoken <tu authtoken>)
ngrok http 8000

# Cloudflare Quick Tunnel (no necesita cuenta; la URL cambia cada vez que lo reinicias)
cloudflared tunnel --url http://localhost:8000

# Tailscale Funnel (con sesión iniciada en tu tailnet)
tailscale funnel 8000
```

Cada herramienta imprime la URL pública al arrancar.

## 3. Las 5 pruebas con curl (antes del dry run)

Se corren contra la URL pública, desde **otra red** (laptop o celular con datos móviles). Crea `request.json` con un item de `public_sample.json`:

```json
{
  "id": "sample-01",
  "context": "BERT (Bidirectional Encoder Representations from Transformers) was introduced by researchers at Google in 2018. Unlike left-to-right language models, BERT is pre-trained to use context from both directions at every layer. Two model sizes were released: BERT-base, with 12 layers, a hidden size of 768 and 12 attention heads, and BERT-large, with 24 layers, a hidden size of 1024 and 16 attention heads. Pre-training used the BooksCorpus and English Wikipedia and combined two objectives: masked language modeling and next sentence prediction.",
  "question": "How many attention heads does BERT-large have?"
}
```

| # | Prueba | Esperado |
|---|---|---|
| 1 | `/health`, sin token | `200 {"status": "ok"}` |
| 2 | `/info`, sin token | `401` |
| 3 | `/info`, token incorrecto | `401` |
| 4 | `/info`, token válido | `200` con `team`, `model`, `params` |
| 5 | `/predict`, token válido | `200` con `id`, `answer`, `start`, `score` |

**Linux / macOS (bash o zsh):**
```bash
export QA_API_TOKEN=<token>

curl -i https://<public-url>/health
curl -i https://<public-url>/info
curl -i -H "Authorization: Bearer wrong" https://<public-url>/info
curl -i -H "Authorization: Bearer $QA_API_TOKEN" https://<public-url>/info
curl -i -X POST https://<public-url>/predict \
  -H "Authorization: Bearer $QA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @request.json
```

**Windows (PowerShell; usa `curl.exe`, no `curl`):**
```powershell
$env:QA_API_TOKEN = "<token>"

curl.exe -i https://<public-url>/health
curl.exe -i https://<public-url>/info
curl.exe -i -H "Authorization: Bearer wrong" https://<public-url>/info
curl.exe -i -H "Authorization: Bearer $env:QA_API_TOKEN" https://<public-url>/info
curl.exe -i -X POST https://<public-url>/predict `
  -H "Authorization: Bearer $env:QA_API_TOKEN" `
  -H "Content-Type: application/json" `
  -d "@request.json"
```

Para la prueba 5, compara la respuesta con las respuestas de referencia del item en `public_sample.json` (para `sample-01`: `16`, `start` 383).

## 4. Dry run y presentación

- El dry run se hace **por el túnel, no por `localhost`**. Manda la URL pública a Alfonso antes de tu turno del dry run y otra vez antes de tu turno de la presentación (y de nuevo si cambia).
- Para el dry run adjunta una captura de cada prueba funcionando.
- Mantén la laptop **enchufada y con la suspensión desactivada**, y la API y el túnel corriendo toda la sesión.
- El día de la presentación Alfonso da un **token distinto**: ponlo en `QA_API_TOKEN` y reinicia la API antes de que empiece la evaluación.
- Último recurso: si ningún túnel funciona, su evaluador corre sobre `localhost` de tu laptop con un test set cifrado.

## 5. Problemas comunes

| Síntoma | Qué hacer |
|---|---|
| `QA_API_TOKEN is not set` | Defínelo en esa misma terminal y vuelve a correr `run.ps1` / `run.sh` |
| `model.safetensors not found` | El modelo no quedó directo en `weights/modelo-final-v3/` |
| `401` con el token correcto | Reinicia la API: el token se lee solo al arrancar |
| Puerto 8000 ocupado | Cierra el otro programa que lo usa |
| En PowerShell `curl` da errores raros | Usa `curl.exe` |
