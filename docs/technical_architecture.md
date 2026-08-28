```markdown
# Technical Architecture & Engineering Deep-Dive

**Document Version:** 1.0  
**Architecture Pattern:** Event-Driven Async Microservice  
**Author:** Saswata Ghosh  

---

## 1. System Request Lifecycle
To understand the architecture, you must trace a single `GET /predictions/churn/cus_mock_5` request from network to memory and back.

1. **HTTP Binding:** Uvicorn receives the TCP packet and passes it to FastAPI's Starlette router.
2. **Pydantic Validation:** The path parameter `{customer_id}` is validated (must be a string).
3. **Dependency Injection (`get_db`):** FastAPI requests a connection from the SQLAlchemy `AsyncSession` pool.
4. **Async I/O Execution:** `feature_engineering.py` fires a raw SQL `GROUP BY` query to Postgres via `asyncpg`. **The event loop is released.** Uvicorn can now handle other incoming HTTP requests while waiting for the DB disk I/O.
5. **Feature Ordering:** The SQL tuple result is mapped to a Python list, strictly ordered by `feature_columns.json` to prevent model input misalignment.
6. **In-Memory Inference:** The list is passed to `churn_service.predict_churn_risk()`. The `StandardScaler` transforms the data in RAM, and `RandomForest.predict_proba()` executes in RAM. **Zero network calls occur here.**
7. **Serialization:** The resulting Python dict is validated against the `PredictionResponse` Pydantic schema and serialized to JSON.
8. **Connection Release:** The `finally` block in `get_db` returns the connection to the pool. HTTP 200 is dispatched.

---

## 2. The Async Architecture (Why `asyncpg`?)

A common misconception is that FastAPI makes *all* Python code fast. It does not. If you use standard `psycopg2` (a synchronous driver), the database call *blocks the entire Uvicorn worker thread*. If 100 users request predictions, they wait in a single-file line.

**Our Implementation:**
* **Driver:** `asyncpg` (a C-extension written specifically for Postgres async I/O).
* **ORM Wrapper:** SQLAlchemy 2.0 `AsyncSession`.
* **The Trade-off:** Async code is harder to write and debug (you must `await` everything). However, for an I/O bound application like a database-querying API, it increases throughput by 10x-100x on a single CPU core by eliminating context-switching overhead.

---

## 3. MLOps: The "Bake vs. Serve" Paradigm

A major anti-pattern in junior MLOps is having a production API query an MLflow server to download a model at runtime. 

### The Training Phase (Track)
During `ml_pipeline/train_model.py`, we use MLflow strictly as a **tracking server**.
* Logs parameters (`n_estimators=100`).
* Logs metrics (`accuracy=0.8368`).
* Logs the artifact (the `.pkl` file) to MLflow's local blob storage.

### The Baking Phase (Extract)
We run `save_for_prod.py`. This script reaches into MLflow, grabs the latest version of the model, and saves it as a local file (`app/ml_models/model.pkl`). 

### The Serving Phase (Deploy)
The FastAPI Dockerfile runs `COPY ./app ./app`. The `.pkl` file is now baked into the Docker image. 
At API startup (`lifespan` event), we use `joblib.load()` to load the model directly into RAM. 

**Why?**
1. **Latency:** Loading from local disk into RAM takes ~0.1 seconds. Querying MLflow over HTTP takes 2-5 seconds per startup.
2. **Fault Tolerance:** If MLflow crashes, or if the API is deployed to an environment without network access to the MLflow server, the API continues to function perfectly.
3. **Reproducibility:** A specific Docker image tag (e.g., `v1.2`) is permanently tied to a specific `.pkl` file. You can rollback the API container without rolling back the MLflow server.

---

## 4. Infrastructure & Containerization

### The Dockerfile Strategy
We use a single-stage build optimized for Python dependencies.
```dockerfile
FROM python:3.11-slim  # Minimal attack surface, fast pull
RUN apt-get update && apt-get install -y libpq-dev gcc # Required to compile asyncpg C-extensions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt # --no-cache reduces image size by ~200MB
COPY ./app ./app
```
*Note: We deliberately exclude the `ml_pipeline`, `data`, and `notebooks` folders from the image via `.dockerignore`. The production container should only contain serving code.*

### Docker Compose Orchestration
```yaml
services:
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data # Persists data even if container goes down
  api:
    build: .
    environment:
      - DATABASE_URL=postgresql+asyncpg://... # Note the internal DNS 'db', not 'localhost'
    depends_on:
      - db
```
**Internal DNS:** Inside Docker, `localhost` refers to the container itself. To talk to the database, the API must resolve the hostname `db` (the name of the service in the compose file).

---

## 5. Engineering Incident Reports (Debugging Log)

To demonstrate real-world production readiness, here are the three major infrastructure bugs encountered during development and their root causes.

### Incident 1: The `+asyncpg` Driver Collision
* **Symptom:** The ETL script (`generate_stripe_data.py`) crashed with `invalid dsn: missing "=" after "postgresql+asyncpg://..."`.
* **Root Cause:** The `.env` file contained `DATABASE_URL=postgresql+asyncpg://...`. The ETL script uses `psycopg2`. `psycopg2` is a legacy driver that strictly adheres to standard Postgres URI formats and crashes on SQLAlchemy's `+asyncpg` dialect modifier.
* **Resolution:** Implemented a dynamic string strip in the ETL script: `os.getenv("DATABASE_URL").replace("+asyncpg", "")`.

### Incident 2: The Scikit-Learn Version Mismatch
* **Symptom:** Docker container crashed on startup with `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.9.0 when using version 1.3.2`.
* **Root Cause:** The local training environment installed the newest `scikit-learn` (1.9.0) via pip, which serialized the `.pkl` using its updated pickle protocol. The `requirements.txt` initially pinned `scikit-learn==1.3.2`. The Docker container downloaded 1.3.2, which lacked the forward-compatibility to read the newer pickle file.
* **Resolution:** Pinned `requirements.txt` to `scikit-learn==1.9.0` to match the local training environment. *Lesson: Always pin your ML libraries exactly across training and serving environments.*

### Incident 3: The Docker `localhost` Network Hang
* **Symptom:** API container started but hung infinitely at `Waiting for application startup...` with no error.
* **Root Cause:** The `churn_service.py` attempted to query `http://localhost:5000` (MLflow) to find the latest model. Inside the Linux network namespace of the API container, port 5000 was empty. Python's `requests` library initiated a TCP SYN handshake and waited for a response that would never come (a silent timeout).
* **Resolution:** Removed the runtime MLflow query entirely, relying 100% on the pre-baked `.pkl` files (as designed in the MLOps paradigm above).

---

## 6. API Security & Extensibility

### Current State
* **Validation:** Strict Pydantic schemas prevent malformed data from hitting the database or model.
* **CORS:** Wide open (`allow_origins=["*"]`) to allow the local Streamlit frontend to communicate with it.

### Production Readiness Checklist (If deployed to AWS/GCP)
1. **Authentication:** Add `fastapi.security.HTTPBearer` to the `predictions` router. The Streamlit app would pass a Bearer token.
2. **Rate Limiting:** Implement `slowapi` to prevent a single client from spamming the `/predictions/` endpoint and overloading the DB.
3. **HTTPS:** Terminate SSL/TLS at a load balancer (AWS ALB/Nginx) before forwarding traffic to the Docker container.
4. **DB Connection Pooling:** Tune the `pool_size` and `max_overflow` parameters in `create_async_engine` to handle sudden traffic spikes without overwhelming Postgres.
```