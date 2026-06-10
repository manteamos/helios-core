# SYSTEM ARCHITECTURE & ENGINEERING SPECIFICATION: NEXT-GEN PV SIMULATION ENGINE

> Canonical requirements document. CLAUDE.md (repo root) translates this into the
> phased execution plan. Where the two conflict, CLAUDE.md wins (it contains
> deliberate corrections — e.g., HTTP 202 vs. the "222" typo in §4.1 below).

## ROLE & CONTEXT
Design the complete software architecture, database schema, API interface, and core
mathematical solvers for a next-generation, cloud-native Photovoltaic (PV) Yield
Simulation Engine meant to entirely replace legacy desktop applications like PVsyst.
The software must be headless, infinitely scalable, cryptographically secure, and
transition from empirical, static approximations to dynamic, continuous
multi-physics solvers.

---

## SECTION 1: ARCHITECTURAL PARADIGM & SYSTEM TOPOLOGY

Microservices-based, cloud-native architecture optimized for headless Linux
containers orchestrated via Kubernetes.

1. Detailed ASCII system topology diagram mapping:
   - API Gateway / Ingress Layer (FastAPI / gRPC)
   - Message Broker / Task Queue (Celery / RabbitMQ or Redis) for distributed execution
   - Stateless Simulation Compute Worker Nodes (Python/C++ extensions)
   - Cryptographically Verified Component Registry Data Store (PostgreSQL JSONB + Redis cache)
   - Object Storage Layer (S3) for caching multi-gigabyte spatial raster datasets
     and raw time-series output
2. Exact multi-threading and concurrency strategy for sub-hourly, 8760-row or
   minute-level annual simulations across utility-scale portfolios without blocking
   the asynchronous event loop.

---

## SECTION 2: MATHEMATICAL SOLVERS & CORE PHYSICS IMPLEMENTATION

Complete Python implementations (NumPy/SciPy). Production-grade, highly vectorized,
robustly commented.

### 2.1 Continuous Perez Transposition Engine
- Continuous variation of the Perez-Ineichen sky model.
- CRITICAL: eliminate all discrete empirical category lookups or step-change bins.
  Use mean-preserving quadratic or cubic splines to represent the empirical
  parameter space (F11, F12, ...) as a fully continuous function with continuous
  first derivative, enabling robust bisection/optimization during reverse
  transposition.
- Include dynamic circumsolar diffuse, isotropic background, and horizon band
  components.

### 2.2 3-Node Transient Thermal Network Solver
- Replace static steady-state Uc/Uv with an explicit multi-node transient energy
  balance solver.
- Three material nodes: front glass (Tg), silicon cell (Tc), backsheet (Tb).
- Convective coefficients (h_front, h_rear) scale dynamically with continuous 3D
  wind vectors and per-module spatial exposure factors.
- Explicit finite difference over Δt using layer-specific heat capacities (Cp) and
  densities.

### 2.3 Stochastic Cell-Level Mismatch & Lifecycle Aging Matrix
- Monte Carlo engine over an electrical matrix of cells (12 × 6 per module).
- Spatial thermal gradients across the physical rack (corner modules cool faster
  than trapped-heat center modules).
- Arrhenius-driven degradation over a 25-year lifecycle; localized acceleration
  tracks cell temperature non-linearly (Ea = 45 kJ/mol).
- Electrical mismatch: continuous, compounding series-string current bottleneck
  (string current governed by the weakest degraded/soiled cell).

---

## SECTION 3: COMPONENT DATA MODEL & SECURE REGISTRY SPECIFICATION

Zero-trust verification to eradicate banking risk from user-manipulated or
unverified manufacturer parameter files (.PAN / .OND).

1. Strict Pydantic v2 schema for a Verified PV Module Component profile:
   - Full legal manufacturer and model identification
   - Physical layer geometries, cell electrical layout topologies, STC coefficients
   - Low-light parameters and polynomial IAM coefficients
   - `itl_identifier` (Independent Testing Laboratory) and cryptographic
     `digital_signature` fields
2. Validation method re-verifying nominal STC power equals Impp × Vmpp within
   ±0.1%, raising clean exceptions on failure.
3. Asymmetric cryptographic verification (Ed25519 public keys) proving the payload
   was signed by an accredited lab before ingestion into the simulation database.

---

## SECTION 4: PROGRAMMATIC INTERFACE, CONTAINERIZATION, & AUTOMATION RUNTIME

1. FastAPI: asynchronous endpoint `POST /api/v1/simulations/run` accepting project
   configuration metadata, weather time-series, and cryptographic module
   parameters; dispatches to a background worker queue; returns HTTP 202 Accepted
   (spec originally said "222" — typo) with a unique task execution ID.
2. Enterprise Dockerfile: production-hardened, non-root, `python:3.11-slim`,
   stripped of graphical/desktop dependencies, multi-stage build minimizing image
   size and CVE footprint.

---

## OUTPUT EXPECTATIONS
- No hand-waving summaries or placeholder comments (`# TODO: implement math here`).
- Complete mathematical steps, matrix conversions, and functional syntax.
- Explicit markdown formatting and clear parameter naming throughout.
