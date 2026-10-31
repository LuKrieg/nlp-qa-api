# API HANDOFF — Integración, pruebas y despliegue de DistilBERT V2

## 1. Estado del proyecto

Se implementó una API de **Question Answering (QA) extractivo** con **FastAPI**, conectada a **DistilBERT V2**, ajustado mediante *fine-tuning* sobre **SQuAD 2.0**. La inferencia se ejecuta localmente en **CPU**, sin depender de un servicio comercial de modelos.

**Repositorio:** https://github.com/LuKrieg/nlp-qa-api  
**Rama:** `camila/parte3API`  
**Último commit publicado antes de esta actualización documental:** `82c4fb4` (`Document API deployment and handoff`).  
**Estado:** 33/33 verificaciones funcionales locales superadas; cinco ejemplos públicos ejecutados mediante HTTPS; **4/5 respuestas correctas** y **5/5 solicitudes por debajo de 5 segundos**. No se ha integrado la rama con `main`.

> **Alcance de las pruebas:** Los 33 PASS verifican aspectos funcionales y de contrato. No equivalen a un 100 % de exactitud del modelo. Los cinco ejemplos públicos no predicen con certeza los resultados sobre las 50 preguntas ocultas.

## 2. Archivos principales

| Archivo o carpeta | Función |
|---|---|
| `train.py` | Fine-tuning, evaluación y calibración del umbral de no-respuesta |
| `inference.py` | Inferencia CPU, ventanas deslizantes, selección de spans, offsets y respuestas `null` |
| `api.py` | API FastAPI, endpoints y autenticación Bearer |
| `test_api.py` | Pruebas funcionales y de integración locales |
| `requirements.txt` | Dependencias Python |
| `qa_api_contract.yaml` | Contrato OpenAPI proporcionado para el proyecto |
| `INFERENCE_HANDOFF.md` | Instrucciones del módulo de inferencia |
| `API_HANDOFF.md` | Este documento de transferencia |
| `logs/modelo-final-v2/metrics.json` | Métricas y umbral calibrado del modelo V2 |
| `logs/modelo-final-v2/train_history.json` | Historial de entrenamiento |
| `logs/modelo-final-v2/loss.png` | Gráfica de pérdida de entrenamiento y validación |
| `weights/modelo-final-v2/` | Pesos y archivos del tokenizador (distribuidos aparte) |

Los pesos **no están incluidos en GitHub**. Se entregan por separado mediante `modelo-final-v2.zip`. No subir este ZIP ni `model.safetensors` al repositorio.

## 3. Preparar el entorno (Windows PowerShell)

Para obtener esta rama por primera vez:

```powershell
git clone https://github.com/LuKrieg/nlp-qa-api.git
cd nlp-qa-api
git switch camila/parte3API
```

Si la rama aún no figura en el clon local, ejecutar `git fetch origin` y luego `git switch --track origin/camila/parte3API`.

Crear el entorno virtual e instalar las dependencias:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

El desarrollo y las pruebas se realizaron con Python 3.13. Si se usa otra versión, comprobar la compatibilidad de las dependencias.

### Instalar los pesos

Extraer `modelo-final-v2.zip` de modo que la estructura final sea:

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

No duplicar niveles de carpetas como `weights/modelo-final-v2/modelo-final-v2/`. `inference.py` utiliza los pesos de V2 y lee el umbral de `logs/modelo-final-v2/metrics.json`. Mantener ambos archivos disponibles.

## 4. Ejecutar la API localmente

En la **primera terminal**, configurar el token y el nombre del equipo, utilizando valores reales en lugar de los marcadores:

```powershell
$env:QA_API_TOKEN = "REEMPLAZAR_POR_TOKEN_SEGURO"
$env:TEAM_NAME = "REEMPLAZAR_POR_NOMBRE_REGISTRADO"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

La terminal debe permanecer abierta. `TEAM_NAME` debe coincidir con la identificación oficial del equipo. Si no se configura, `api.py` devuelve el valor predeterminado `team-03`, **todavía no confirmado como el nombre oficial**.

En una **segunda terminal**, comprobar el servidor:

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Respuesta observada durante las pruebas: `HTTP 200` y `{"status":"ok"}`.

> Si aparece `WinError 10048`, hay otro proceso ocupando el puerto 8000. Identificarlo antes de detenerlo. Un `404` al visitar `/` no es un problema: no se definió una página principal; usar los endpoints específicos.

## 5. Contrato de la API

| Endpoint | Método | Autenticación | Comportamiento comprobado |
|---|---|---|---|
| `/health` | GET | No | Responde `200` con `{"status":"ok"}` cuando está lista |
| `/info` | GET | Bearer | Devuelve identificación del equipo y metadatos del modelo |
| `/predict` | POST | Bearer | Devuelve `id`, `answer`, `start` y `score` |

Las solicitudes protegidas usan el encabezado `Authorization: Bearer <token>`. El token válido se obtiene de la variable de entorno `QA_API_TOKEN`; no debe incluirse dentro del código.

**Ejemplo de solicitud `/predict`:**

```json
{
  "id": "q-001",
  "context": "BERT was introduced in 2018.",
  "question": "When was BERT introduced?"
}
```

**Respuesta real obtenida:**

```json
{
  "id": "q-001",
  "answer": "2018",
  "start": 23,
  "score": 0.7573
}
```

`start` es el índice de carácter basado en cero dentro del contexto; la subcadena correspondiente debe coincidir exactamente con `answer`. Cuando el modelo se abstiene, `answer` y `start` son `null`. El `score` está en `[0,1]`, pero **no es una probabilidad estadísticamente calibrada**.

Características de inferencia: **CPU**, `max_seq_length=384`, `doc_stride=128`, ventanas superpuestas para contextos largos y umbral de no-respuesta obtenido de las métricas de V2.

## 6. Pruebas locales completadas

Para ejecutar `test_api.py`, configurar en su terminal el mismo token con el que arrancó Uvicorn:

```powershell
$env:QA_API_TOKEN = "EL_MISMO_TOKEN_DEL_SERVIDOR"
.\.venv\Scripts\python.exe test_api.py
```

**Resultado observado: `33/33 PASS`.** Se comprobaron los endpoints, la autorización y rechazo de tokens, la validación de entradas, la estructura de las respuestas y las cinco muestras públicas.

**Nota de portabilidad:** En la versión utilizada durante las pruebas, `test_api.py` tiene una ruta absoluta a `public_sample.json` en una carpeta del escritorio del equipo de desarrollo. Si se ejecuta desde otra computadora, actualizar esa ruta o adaptar el script para recibir la ubicación del archivo. Sin el archivo, sus pruebas públicas podrían omitirse.

| Ejemplo | Predicción V2 | ¿Respuesta correcta? | Latencia HTTP local |
|---|---|---|---:|
| `sample-01` | `16` | Sí | 187 ms |
| `sample-02` | `3.5 days` | Sí | 154 ms |
| `sample-03` | `null` | Sí | 139 ms |
| `sample-04` | Respuesta no nula (falso positivo) | No | 150 ms |
| `sample-05` | `50,257` | Sí | 472 ms |

Resultado: **4/5 respuestas públicas correctas**. El caso `sample-04` es una pregunta imposible para la cual el modelo produce una respuesta en lugar de `null`.

## 7. Despliegue HTTPS con Tailscale Funnel

El túnel configurado en la computadora de desarrollo publica el servidor local:

**URL pública utilizada en las pruebas:** https://camila-jovel.tailad9356.ts.net/  
**Destino:** `http://127.0.0.1:8000`

Con Tailscale instalado, conectado a la red autorizada y Uvicorn en ejecución:

```powershell
tailscale funnel --bg 8000
```

Consultar su estado:

```powershell
tailscale funnel status
```

Detener la exposición pública cuando ya no sea necesaria:

```powershell
tailscale funnel --https=443 off
```

Funnel se configuró en la red propia de la computadora de desarrollo. La URL puede dejar de estar disponible si Uvicorn se detiene, el equipo se apaga o se desconecta, o se deshabilita Funnel. **No debe considerarse alojamiento permanente**. Otro integrante puede crear un túnel propio y obtener una URL diferente.

### Verificaciones reales por HTTPS

| Prueba | Resultado observado |
|---|---|
| `GET /health` sin token | `200 OK`, `{"status":"ok"}` |
| `GET /info` sin token | `401 Unauthorized`, `{"error":"unauthorized"}` |
| `GET /info` con token válido | `200 OK` con metadatos del modelo |
| `POST /predict` con token válido | Respuesta extractiva correcta y campos completos |
| Solicitud de ejemplo corto por HTTPS | 215.15 ms; `answer="2018"`, `start=23` |

### Pruebas completas de `public_sample.json` mediante HTTPS

Se enviaron las cinco solicitudes de `public_sample.json` al endpoint HTTPS con autenticación Bearer. Se registró el tiempo transcurrido de cada solicitud, su coincidencia con las referencias y la validez del offset.

| Ejemplo | Respuesta esperada/resultado | ¿Correcto? | ¿Offset válido? | Latencia HTTPS | ¿Menos de 5 s? |
|---|---|---|---|---:|---|
| `sample-01` | `16` | Sí | Sí | **185.5 ms** | Sí |
| `sample-02` | `3.5 days` | Sí | Sí | **149.5 ms** | Sí |
| `sample-03` | `null` | Sí | Sí | **142.5 ms** | Sí |
| `sample-04` | Respuesta no nula; debería ser `null` | **No** | Sí | **137.8 ms** | Sí |
| `sample-05` | `50,257`; contexto de más de 512 tokens | Sí | Sí | **442.9 ms** | Sí |

**Resumen de esta ejecución:**

- **Exactitud:** 4/5 (**80 %**) en los ejemplos públicos.
- **Offsets válidos:** 5/5; en el caso imposible con falso positivo, el offset de la cadena predicha sigue siendo válido, aunque la respuesta sea semánticamente incorrecta.
- **Límite individual de 5 segundos:** 5/5 solicitudes dentro del límite.
- **Latencia promedio HTTPS:** **211.6 ms**.
- **Latencia máxima HTTPS:** **442.9 ms** (`sample-05`, contexto largo).
- **Suma de tiempos medidos para las cinco solicitudes:** **1,058.2 ms (1.06 s)**.

La guía establece **5 segundos por solicitud** y **150 segundos para el conjunto de evaluación**. Los resultados anteriores cumplen estos límites **solo para las cinco muestras públicas evaluadas**; no permiten garantizar tiempos ni exactitud en las 50 preguntas ocultas.

## 8. Seguridad y mantenimiento

- Utilizar `QA_API_TOKEN` mediante variables de entorno; nunca incluir el valor real en GitHub, capturas compartidas o documentos públicos.
- **Cambiar el token temporal de las pruebas antes de cualquier evaluación pública**, porque un token de demostración expuesto no debe reutilizarse como secreto de producción.
- Reiniciar Uvicorn si cambia `QA_API_TOKEN` o `TEAM_NAME` en el entorno del proceso del servidor.
- Confirmar el identificador oficial del equipo en `/info` antes de la entrega.
- Conservar las respuestas y los códigos HTTP exigidos por `qa_api_contract.yaml`.
- Evitar subir los archivos de pesos, el ZIP y los logs temporales o datos innecesarios a Git.
- Cerrar Funnel cuando no se necesite acceso desde internet.
- No crear la etiqueta `submission` hasta revisar el conjunto completo de entregables con el equipo.

## 9. Trabajo pendiente para el integrante 4

**Pruebas ya completadas; no repetirlas salvo cambios en el código, el modelo, el hardware o el despliegue.** Las cinco muestras públicas por HTTPS y sus latencias están registradas en la sección 7.

**Pendientes reales:**

1. Confirmar el nombre oficial del equipo y configurar `TEAM_NAME`.
2. Preparar el token final para la evaluación, reiniciar el servidor con esa configuración y verificar autenticación.
3. Configurar y probar un túnel HTTPS alternativo como respaldo, si la guía lo requiere.
4. Completar `REPORT.md` a partir de `REPORT_TEMPLATE.md`, incorporando las métricas de V2, la calibración del no-answer, los resultados HTTPS, las limitaciones y la comparación con decoders.
5. Integrar los cambios de las ramas necesarias mediante la revisión acordada por el equipo; `main` no ha sido modificado por estas publicaciones.
6. Verificar que el evaluador pueda acceder a los pesos y al servicio durante la demostración.
7. Preparar el pitch de tres minutos y las respuestas individuales a preguntas técnicas.
8. Realizar una prueba general (*dry run*) con el despliegue y el token finales, y crear la etiqueta `submission` únicamente cuando corresponda.

## 10. Limitaciones conocidas

1. **Falsos positivos en preguntas imposibles:** `sample-04` devuelve una respuesta cuando debería devolver `null`.
2. **Generalización no verificada sobre las 50 preguntas ocultas:** cinco ejemplos públicos son insuficientes para estimar con precisión el resultado final.
3. **Latencia dependiente de la longitud:** contextos largos necesitan más ventanas; aunque `sample-05` tardó 442.9 ms por HTTPS, otros textos podrían tardar más.
4. **Disponibilidad del servicio:** el enlace HTTPS solo funciona mientras el servidor local, el equipo y el túnel permanezcan operativos.
5. **Interpretación del score:** el valor de `[0,1]` proviene de una transformación del margen de decisión; no equivale a una confianza probabilística calibrada.
6. **Portabilidad del script de pruebas:** `test_api.py` usa una ruta local absoluta para el conjunto público; debe adaptarse antes de ejecutarlo en otra computadora.

---

**Estado al cierre de estas pruebas:** entrenamiento V2, inferencia CPU, API autenticada y túnel HTTPS implementados. **33/33 comprobaciones locales exitosas; 4/5 respuestas públicas correctas; 5/5 offsets válidos; 5/5 tiempos HTTPS por debajo de 5 s.** Código y documentación en la rama `camila/parte3API`. Pendientes la configuración definitiva para evaluación y los entregables finales del equipo.
