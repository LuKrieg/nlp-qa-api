# Entrega del Módulo de Inferencia (Handoff para Integrante 3)

## 1. Módulo e Inicialización
El módulo `inference.py` expone la clase `QAInference`.
- **Rutas portables:** Resueltas automáticamente hacia `weights/modelo-final-v2/` y `logs/modelo-final-v2/metrics.json`.
- **Hardware:** Inferencia forzada en CPU (`torch.device('cpu')`) con `torch.inference_mode()`.
- **Ventanas:** Longitud `max_length=384`, superposición `stride=128`.
- **Umbral no-respuesta:** $\tau = 1.5017$ calibrado en validación.

```python
from inference import QAInference

qa = QAInference()
result = qa.predict(context="...", question="...")
```

## 2. Firma, Validaciones y Retorno
- **Firma:** `predict(context: str, question: str) -> Dict[str, Any]`
- **Validaciones:** Lanza `ValueError` si las entradas no son cadenas, están vacías o violan los límites del contrato ($1 \le \text{context} \le 5000$, $1 \le \text{question} \le 500$).
- **Campos devueltos:**
  - `answer` (`str` o `None` si se abstiene).
  - `start` (`int` 0-based o `None` si `answer` es `None`). Cumple el invariante `context[start:start+len(answer)] == answer`.
  - `score` (`float` en $[0.0, 1.0]$), confianza calculada según la distancia al umbral.

> **Importante para la API:** `predict()` no recibe ni retorna `id`. El servicio HTTP debe capturar el `id` del request y anexarlo tal cual al JSON de respuesta.

## 3. Resultados del Benchmark Público (CPU)
- **Aciertos:** 4 de 5 ejemplos correctos (`sample-01`, `sample-02`, `sample-03` y `sample-05` con >512 tokens).
- **Fallo conocido:** `sample-04` (pregunta imposible donde el modelo genera un falso positivo espurio).
- **Latencia CPU:** ~160–250 ms en textos estándar; ~540 ms en textos multiventana (>512 tokens). Ampliamente dentro del límite de 5 s por request.

## 4. Distribución de Pesos y Tareas para la API
- **Pesos:** No subir la carpeta `weights/` a Git. Descargar `modelo-final-v2` desde el GitHub Release del equipo o la carpeta compartida designada.
- **Implementación API (`qa_api_contract.yaml`):**
  - Servidor en puerto 8000.
  - `GET /health`: Público; 200 `{"status": "ok"}` (o 503 mientras carga).
  - `GET /info`: Protegido; 200 con `team`, `model`, `params`, `max_seq_length`, `doc_stride`.
  - `POST /predict`: Protegido; validar token Bearer contra variable de entorno `QA_API_TOKEN`.
  - Manejo uniforme de errores: `{"error": "<code>", "detail": "..."}` con códigos `unauthorized` (401), `bad_request` (400/422) e `internal_error` (500).

