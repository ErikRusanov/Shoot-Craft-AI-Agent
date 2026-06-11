# Business flow

End-to-end journey from user photos to delivered result.

```mermaid
flowchart TD
    A([User photos]) --> B["POST /v1/faces/{face_key}\nIngest & quality gate"]

    B --> C{Gate verdict}
    C -- PASSED --> D
    C -- "SOFT\n(usable with risk)" --> E{Business\ndecision}
    C -- BELOW_FLOOR --> F([Reject — re-shoot])

    E -- accept --> D
    E -- reject --> F

    D["POST /v1/sessions/{session_key}\nStart session"] --> G["GET /events\nSSE stream"]

    G --> H{Preset has\nask slot?}
    H -- yes --> I["NeedInputEvent\n→ POST /input\n(free-form scene)"]
    H -- no --> J

    I --> J["PlanEvent + CostEvent\n(estimate)"]
    J --> K{Approve?}
    K -- "POST /approve approved=false" --> L([FAILED — plan_rejected])
    K -- "POST /approve approved=true" --> M

    M["Generation loop\ngenerate → face-check → keep-best"]
    M --> N{Identity\nconverged?}
    N -- "similarity ≥ threshold" --> O["ResultEvent\n(best frame)"]
    N -- "budget spent,\nno deliverable" --> P([FAILED — budget / no_deliverable])
    N -- retry --> M

    O --> Q([DoneEvent\nSSE closes])
```

## Key invariants

| Rule | Why |
|------|-----|
| Face key is scoped to one identity; session key scopes one shoot | Allows retries without re-uploading the photo |
| Every mutation carries `idem_key` | Guarantees at-most-once execution across network retries |
| Budget is the number of **paid** generator calls, not iterations | Caller controls spend ceiling; the loop exhausts it before declaring failure |
| Best result is kept even if the loop fails to converge | Caller always gets something if at least one frame passed SOFT |
