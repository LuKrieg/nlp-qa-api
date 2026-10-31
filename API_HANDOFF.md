# API HANDOFF — Integración y despliegue de DistilBERT V2

## 1. Estado actual del proyecto

La API de Question Answering (QA) fue desarrollada utilizando FastAPI e integrada con el modelo DistilBERT V2, ajustado mediante Fine-Tuning con SQuAD 2.0.

**Estado:** Implementación funcional, con 33/33 pruebas locales superadas y conexión HTTPS verificada mediante Tailscale Funnel.

**Repositorio:** https://github.com/LuKrieg/nlp-qa-api

**Rama de trabajo:** `camila/parte3API`

**Último commit de la API:** `8df6677`

La rama incluye el entrenamiento, el módulo de inferencia en CPU y la implementación de FastAPI. No se ha realizado merge con `main`.

## 2. Archivos principales

| Archivo | Función |
|---|---|
| `train.py` | Entrenamiento y Fine-Tuning del modelo |
| `inference.py` | Inferencia en CPU, ventanas deslizantes, offsets y no-answer |
| `api.py` | Servicio FastAPI, autenticación y endpoints |
| `test_api.py` | Pruebas automatizadas del servicio |
| `requirements.txt` | Dependencias del proyecto |
| `qa_api_contract.yaml` | Contrato OpenAPI oficial |
| `INFERENCE_HANDOFF.md` | Documentación del módulo de inferencia |
| `logs/modelo-final-v2/` | Métricas, historial y gráfica de entrenamiento |

Los pesos se distribuyen por separado en `modelo-final-v2.zip`.

## 3. Preparación del entorno (Windows PowerShell)

Obtener el repositorio y seleccionar la rama:

```powershell
git clone https://github.com/LuKrieg/nlp-qa-api.git
cd nlp-qa-api
git switch camila/parte3API
```

Si el repositorio ya está clonado, basta con actualizar la referencia remota y seleccionar la rama correspondiente.

Crear el entorno virtual:

```powershell
py -3.13 -m venv .venv
```

Instalar dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se requiere Python compatible con las dependencias del proyecto. El entorno utilizado durante el desarrollo fue Python 3.13.

## 4. Instalación de los pesos

El archivo `modelo-final-v2.zip` se comparte externamente por WhatsApp. No está almacenado en GitHub.

Extraer el contenido y comprobar que exista esta estructura:

```text
nlp-qa-api/
├── api.py
├── inference.py
├── test_api.py
├── logs/
│   └── modelo-final-v2/
│       └── metrics.json
└── weights/
    └── modelo-final-v2/
        ├── model.safetensors
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── ...
```

Evitar crear una carpeta duplicada `modelo-final-v2/modelo-final-v2`.

El modelo se carga desde `weights/modelo-final-v2/` y su umbral calibrado desde `logs/modelo-final-v2/metrics.json`.

## 5. Ejecutar la API localmente

Configurar las variables de entorno antes de iniciar el servidor:

```powershell
$env:QA_API_TOKEN = "REEMPLAZAR_POR_TOKEN_DE_DESARROLLO"
$env:TEAM_NAME = "REEMPLAZAR_POR_NOMBRE_REGISTRADO"
```

**Importante:** el nombre del equipo debe coincidir exactamente con el registrado ante el profesor. `team-03` es únicamente el valor predeterminado del código y no está confirmado como identificador definitivo.

Iniciar FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Mantener esta terminal abierta mientras el servicio se encuentre en uso.

Comprobar:

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{"status":"ok"}
```

La documentación interactiva de FastAPI puede consultarse en `http://127.0.0.1:8000/docs`, si permanece habilitada.

## 6. Endpoints implementados

El servicio sigue las especificaciones de `qa_api_contract.yaml`.

| Endpoint | Método | Autenticación | Resultado |
|---|---|---|---|
| `/health` | GET | Pública | 200 `ok`, o 503 `loading` |
| `/info` | GET | Bearer obligatorio | Información del modelo |
| `/predict` | POST | Bearer obligatorio | Respuesta extractiva o `null` |

**Formato de entrada para `/predict`:**

```json
{
  "id": "q-001",
  "context": "BERT was introduced in 2018.",
  "question": "When was BERT introduced?"
}
```

**Ejemplo de respuesta obtenida:**

```json
{
  "id": "q-001",
  "answer": "2018",
  "start": 23,
  "score": 0.7573
}
```

La respuesta debe ser una subcadena exacta del contexto. Cuando no existe respuesta, `answer` y `start` deben ser `null`.

El servicio utiliza DistilBERT V2 en CPU, con ventanas de longitud máxima de 384 tokens y stride de 128.

La autenticación se realiza mediante `Authorization: Bearer <token>`. El token se lee exclusivamente de la variable de entorno `QA_API_TOKEN`.

## 7. Pruebas locales

En una segunda terminal PowerShell, configurar el mismo token que utiliza el servidor:

```powershell
$env:QA_API_TOKEN = "REEMPLAZAR_POR_EL_MISMO_TOKEN"
.\.venv\Scripts\python.exe test_api.py
```

**Resultado obtenido: 33/33 PASS.**

Las pruebas comprueban disponibilidad, autenticación, validaciones, formato de respuestas y ejecución de los cinco ejemplos públicos.

Resultados con DistilBERT V2:

| Ejemplo | Resultado | Latencia HTTP local |
|---|---|---:|
| sample-01 | Correcto: `16` | 187 ms |
| sample-02 | Correcto: `3.5 days` | 154 ms |
| sample-03 | Correcto: `null` | 139 ms |
| sample-04 | Incorrecto: falso positivo | 150 ms |
| sample-05 | Correcto: `50,257` | 472 ms |

**Precisión en el conjunto público: 4/5 respuestas correctas.**

Las 33 pruebas superadas representan cumplimiento funcional de los casos comprobados, no una precisión de 100 % del modelo.

## 8. Despliegue HTTPS mediante Tailscale Funnel

Se configuró Tailscale Funnel en la laptop utilizada durante el desarrollo.

**URL pública configurada:**

https://camila-jovel.tailad9356.ts.net/

El túnel redirige tráfico HTTPS al servicio local en `http://127.0.0.1:8000`.

Para activar Funnel, con Tailscale conectado y autorizado:

```powershell
tailscale funnel --bg 8000
```

Para consultar su configuración:

```powershell
tailscale funnel status
```

Para detenerlo:

```powershell
tailscale funnel --https=443 off
```

La disponibilidad de esta URL depende de que la laptop, Uvicorn y Funnel continúen funcionando. Otro integrante puede necesitar configurar su propio túnel y utilizar una URL distinta.

**Pruebas HTTPS realizadas:**

- `/health` sin token: 200 OK.
- `/info` sin token: 401 Unauthorized.
- `/info` con token válido: 200 OK.
- `/predict` autenticado: respuesta correcta (`2018`, offset 23).

**Latencia HTTPS medida:** 215.15 ms para una pregunta corta, incluyendo comunicación e inferencia.

El profesor establece un límite de cinco segundos por solicitud y 150 segundos para la evaluación completa.

La prueba HTTPS del contexto largo `sample-05` sigue pendiente.

## 9. Seguridad y restricciones

- No subir los pesos `.safetensors` ni el archivo ZIP a GitHub.
- No guardar tokens reales en el código, documentación ni commits.
- Utilizar `QA_API_TOKEN` como variable de entorno.
- Configurar `TEAM_NAME` con el identificador oficial del equipo.
- Sustituir el token de desarrollo por el asignado para la evaluación.
- Mantener las rutas y respuestas compatibles con `qa_api_contract.yaml`.
- No utilizar servicios comerciales de inferencia. El modelo se ejecuta localmente.
- No crear todavía la etiqueta `submission` hasta que el equipo complete y revise la entrega.

## 10. Pendientes para el integrante 4

- Confirmar el nombre oficial del equipo y configurar `TEAM_NAME`.
- Realizar las cinco pruebas públicas mediante HTTPS, incluyendo el contexto largo.
- Medir la latencia real mediante el túnel y comprobar los límites de evaluación.
- Configurar y verificar un segundo método de túnel como respaldo, según la guía.
- Completar `REPORT.md` con todos los apartados de `REPORT_TEMPLATE.md`.
- Documentar la arquitectura, Fine-Tuning, resultados, umbral de no-respuesta, latencias, comparación con decoders y limitaciones.
- Preparar el pitch de tres minutos y el Q&A individual.
- Coordinar la integración de ramas con el equipo.
- Realizar el dry run y crear la etiqueta `submission` sobre el commit definitivo, siguiendo las instrucciones del profesor.

## 11. Limitaciones conocidas

1. El modelo puede devolver una respuesta incorrecta en preguntas imposibles. `sample-04` es un falso positivo conocido.
2. Las métricas de SQuAD 2.0 y las cinco preguntas públicas no garantizan el rendimiento sobre las 50 preguntas ocultas.
3. Los contextos largos requieren varias ventanas y mayor tiempo de procesamiento.
4. La API depende de que el modelo esté disponible localmente y de que el servicio permanezca ejecutándose.
5. La puntuación `score` se calcula a partir del margen respecto al umbral, pero no representa una probabilidad estadísticamente calibrada.

---

**Estado de entrega:** Entrenamiento, inferencia y API implementados, probados localmente y publicados en la rama `camila/parte3API`. HTTPS funcional con Tailscale Funnel. Pendientes las verificaciones externas completas y los entregables finales del equipo.