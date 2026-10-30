# Entrega del Módulo de Inferencia (Handoff para Integrante 3)

## 1. Setup y Entorno de Trabajo (Windows PowerShell)

1. **Obtener la rama de trabajo:**
   ```powershell
   git fetch origin
   git checkout camila/parte2
   ```

2. **Crear entorno virtual e instalar dependencias:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. **Descarga y extracción de pesos:**
   Descarga `modelo-final-v2.zip` desde el Release oficial o drive compartido. Extráelo en `weights/` asegurando que los archivos queden directamente bajo `weights/modelo-final-v2/` (ej. `weights/modelo-final-v2/model.safetensors`), sin anidar carpetas duplicadas (`weights/modelo-final-v2/modelo-final-v2/`).
   > **ADVERTENCIA:** Nunca hagas commit ni push de archivos `.zip`, binarios de `weights/` ni tokens. Verifica que estén en `.gitignore`.

4. **Verificación local del módulo:**
   ```powershell
   .\.venv\Scripts\python.exe -c "from inference import QAInference; qa = QAInference(); print('OK:', qa.predict('Context test.', 'Test?'))"
   .\.venv\Scripts\python.exe test_regression.py
   ```
   Todas las pruebas deben mostrar `PASS`.

---

## 2. Especificación Técnica de `inference.py`

- **Clase:** `QAInference` (resuelve rutas portables a `weights/modelo-final-v2/` y `logs/modelo-final-v2/metrics.json`).
- **Hardware:** CPU obligatorio (`torch.device('cpu')`, `torch.inference_mode()`).
- **Ventanas:** Longitud `max_length=384`, `stride=128`.
- **Umbral de no-respuesta:** $\tau = 1.5017$ (calibrado en validación).
- **Firma:** `predict(context: str, question: str) -> Dict[str, Any]`
- **Validaciones:** Lanza `ValueError` ante entradas vacías, tipos inválidos o fuera de rango ($1 \le \text{context} \le 5000$, $1 \le \text{question} \le 500$).
- **Retorno:**
  - `answer`: `str` o `None` si se abstiene.
  - `start`: `int` (offset base 0) o `None`. Cumple: `context[start:start+len(answer)] == answer`.
  - `score`: `float` en $[0.0, 1.0]$.
  > **Nota API:** `predict()` no gestiona el `id`. Tu servicio debe capturar el `id` de la solicitud y hacerle *echo* exacto en el JSON de salida.

- **Comportamiento conocido (`public_sample.json`):** 4/5 aciertos. Resuelve exitosamente contextos largos multiventana (`sample-05`). Presenta un falso positivo en la pregunta imposible adversarial `sample-04`. Latencias en CPU: ~160–250 ms (1 ventana) y ~540 ms (3 ventanas).

---

## 3. Tareas de Implementación del Servicio API (`qa_api_contract.yaml`)

El servicio debe escuchar localmente en el puerto 8000:
1. **`GET /health` (Público):** Retorna `200 {"status": "ok"}` cuando el modelo esté listo, o `503 {"status": "loading"}` durante la carga.
2. **Control de Acceso:** Endpoints protegidos requieren `Authorization: Bearer <token>`. Leer el token desde la variable de entorno `QA_API_TOKEN` (`$env:QA_API_TOKEN` en PowerShell). Si falta o no coincide, retornar `401 {"error": "unauthorized"}`.
3. **`GET /info` (Protegido):** Retorna `200` con `team`, `model`, `params` (total del modelo), `max_seq_length` (384) y `doc_stride` (128).
4. **`POST /predict` (Protegido):** Valida JSON entrante (`id`, `context`, `question`). Ante errores de formato retornar `400` o `422` con `{"error": "bad_request", "detail": "..."}`. En caso de error interno retornar `500 {"error": "internal_error"}`.
