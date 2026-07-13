# Product Requirements Document — [Project Name]

**Author:** | **Date:** | **Status:** Draft / Approved

## 1. Problem Statement
What business/operational problem does this solve? Who feels the pain today, and how is it handled now?

## 2. Objective & Success Criteria
One sentence objective. Then measurable success criteria, e.g.:
- Detect [X] with recall ≥ [N]% at precision ≥ [M]% on the acceptance test set
- End-to-end latency ≤ [N] ms per frame on [target hardware]
- System uptime ≥ [N]% during operating hours

## 3. CV Task Mapping
| Business ask | Canonical CV task | Output |
|---|---|---|
| e.g. "flag workers without helmets" | Object detection + tracking | boxes + track IDs + alert events |

## 4. Users & Usage Context
Who consumes the output (operator dashboard, alert system, report)? Frequency, environment (indoor/outdoor, lighting, camera positions), scale (streams, images/day).

## 5. Data Position
- Available data today (source, quantity, labeled?)
- Data to collect (how, by whom, timeline)
- Labeling plan (tool, format, auto-label assist?)
- Privacy/compliance constraints (faces, PII, on-prem requirement?)

## 6. Constraints
- Hardware / deployment target (edge device, server, cloud)
- Latency / throughput budget
- Licensing constraints (e.g., no AGPL in shipped code)
- Budget (GPU hours, labeling, infra)

## 7. Cost of Errors
What happens on a false positive? On a false negative? Which is worse? (This drives threshold tuning and model choice.)

## 8. Scope
**In scope:** …
**Out of scope (v1):** …

## 9. Milestones
| Milestone | Deliverable | Target |
|---|---|---|
| M1 | Zero-shot/pretrained feasibility baseline + eval set | |
| M2 | Labeled dataset v1 + trained model meeting draft bar | |
| M3 | Deployed pilot + monitoring | |

## 10. Open Questions
