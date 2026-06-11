# Architecture

## Layer diagram

Dependencies point strictly inward toward `schemas/` and `protocols/`.

```mermaid
graph TB
    subgraph api ["API layer — src/api/"]
        R[routes.py\nHTTP endpoints]
        S[sse.py\nSSE tail]
        D[deps.py\nDI wiring]
    end

    subgraph graph ["Orchestration — src/graph/"]
        B[builder.py\nLangGraph FSM]
        N[nodes.py\nFSM nodes]
    end

    subgraph services ["Domain logic — src/services/"]
        VS[vision]
        QG[quality_gate]
        GL[generation_loop]
        FC[facecheck]
        PM[preset_matcher]
        PB[prompt_builder]
        BU[budget]
        ID[idempotency]
    end

    subgraph protocols ["Ports — src/protocols/"]
        IG[ImageGenerator]
        FA[FaceAnalyzer]
        EM[Embedder]
        SS[StateStore]
        EB[EventBus]
        OS[ObjectStorage]
        SF[SlotFiller]
    end

    subgraph connectors ["Connectors — src/services/connectors/"]
        OR[openrouter_generator]
        IF[insightface_embedder]
        FK[fake]
        RS[redis_store]
        RE[redis_event_bus]
        MS[memory_store]
        S3[s3_storage]
        LS[local_storage]
        TH[throttle]
    end

    subgraph schemas ["Contracts — src/schemas/"]
        CT[contract.py]
        ST[state.py]
        EV[events.py]
        PR[presets.py]
    end

    R --> B
    S --> EB
    D --> connectors
    D --> B
    B --> N
    N --> services
    services --> protocols
    protocols -.implements.- connectors
    services --> schemas
    api --> schemas
```

## Connector matrix

| Connector | Implements | Active when |
|-----------|-----------|-------------|
| `openrouter_generator` | `ImageGenerator` | `FAKE_CONNECTORS=false` (default) |
| `throttle` | `ImageGenerator` (wrapper) | wraps the real or fake generator |
| `insightface_embedder` | `FaceAnalyzer`, `Embedder` | `FAKE_CONNECTORS=false` |
| `fake` | `FaceAnalyzer`, `ImageGenerator`, `Embedder` | `FAKE_CONNECTORS=true` |
| `redis_store` | `StateStore` | `REDIS_URL` set |
| `redis_event_bus` | `EventBus` | `REDIS_URL` set |
| `memory_store` | `StateStore`, `EventBus` | `REDIS_URL` unset (dev fallback) |
| `s3_storage` | `ObjectStorage` | `OBJECT_STORAGE=s3` |
| `local_storage` | `ObjectStorage` | `OBJECT_STORAGE=local` (default) |
| `openrouter_slot_filler` | `SlotFiller` | `FAKE_CONNECTORS=false` |

## Data flow (one session)

```
POST /v1/faces  →  Vision (FaceAnalyzer port)  →  quality gate  →  FaceProfile → StateStore
POST /v1/sessions  →  preset_matcher  →  background task spawned
  background: FSM runs
    ask node    → interrupt → resume on POST /input
    approve node → interrupt → resume on POST /approve
    generate node → generation_loop
      loop: prompt_builder → ImageGenerator → facecheck → keep-best → budget.charge
      each iteration → EventBus.publish
GET /events  →  SSE tail (EventBus)  →  client
```
