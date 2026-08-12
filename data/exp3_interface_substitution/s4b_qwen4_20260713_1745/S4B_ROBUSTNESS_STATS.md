# S4b Qwen Robustness Analysis

S4b is a targeted open-weight reproducibility check. It asks whether the S4a ordering of the two strongest contrasts is preserved when generation is moved from the commercial main backbone to self-hosted Qwen3-32B executed through a 4-bit bitsandbytes service.

## Generation And Audit

- Outputs completed: 16/16.
- Primary design-gate eligible rows: 13/16.
- Audit issues: 3.

| Arm | n | primary eligible | mean words | mean LLM calls | mean completion tokens |
|---|---:|---:|---:|---:|---:|
| intact | 4 | 4 | 904.250 | 8.000 | 1901.250 |
| o_rank_slab_matched | 4 | 3 | 843.500 | 8.000 | 1848.250 |
| v_guarded_stress | 4 | 4 | 895.500 | 2.000 | 1276.250 |
| v_unguarded_stress | 4 | 2 | 814.250 | 1.000 | 1168.500 |

## Pairability

- O semantic vs rank slab: primary n=3 (3D Gaussian Splatting, Federated Learning, Multimodal Large Language Models); all-ok n=4.
- Guarded V vs unguarded stress: primary n=2 (Federated Learning, Multimodal Large Language Models); all-ok n=4.

## Design-Gate Issues

- 3D Gaussian Splatting / v_unguarded_stress: word_count_out_of_window (words=789).
- Graph Neural Networks / o_rank_slab_matched: word_count_out_of_window (words=781).
- Graph Neural Networks / v_unguarded_stress: word_count_out_of_window (words=795).

## Blind Quality Contrasts

Positive delta means arm A scored higher than arm B. Primary uses design-gate eligible pairs; all-ok is a sensitivity analysis over all successfully generated pairs.

| Eligibility | Contrast | Endpoint | n | A mean | B mean | delta | 95% bootstrap CI | exact p | sign-biserial | W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | O semantic vs rank slab | overall | 3 | 3.639 | 2.750 | 0.889 | [0.500, 1.417] | 0.250 | 1.000 | 3/0/0 |
| primary | O semantic vs rank slab | organizational_quality | 3 | 4.000 | 2.444 | 1.556 | [1.000, 2.000] | 0.250 | 1.000 | 3/0/0 |
| primary | O semantic vs rank slab | critical_synthesis | 3 | 3.111 | 2.667 | 0.444 | [0.000, 1.000] | 0.500 | 1.000 | 2/1/0 |
| primary | O semantic vs rank slab | global_coherence | 3 | 4.000 | 2.667 | 1.333 | [0.667, 2.333] | 0.250 | 1.000 | 3/0/0 |
| primary | O semantic vs rank slab | citation_plausibility | 3 | 3.444 | 3.222 | 0.222 | [-0.667, 1.000] | 0.750 | 0.333 | 2/0/1 |
| primary | Guarded V vs unguarded stress | overall | 2 | 3.625 | 2.167 | 1.458 | [1.333, 1.583] | 0.500 | 1.000 | 2/0/0 |
| primary | Guarded V vs unguarded stress | organizational_quality | 2 | 3.833 | 2.167 | 1.667 | [1.333, 2.000] | 0.500 | 1.000 | 2/0/0 |
| primary | Guarded V vs unguarded stress | critical_synthesis | 2 | 2.833 | 1.667 | 1.167 | [0.333, 2.000] | 0.500 | 1.000 | 2/0/0 |
| primary | Guarded V vs unguarded stress | global_coherence | 2 | 4.000 | 2.000 | 2.000 | [2.000, 2.000] | 0.500 | 1.000 | 2/0/0 |
| primary | Guarded V vs unguarded stress | citation_plausibility | 2 | 3.833 | 2.833 | 1.000 | [0.333, 1.667] | 0.500 | 1.000 | 2/0/0 |
| all_ok | O semantic vs rank slab | overall | 4 | 3.729 | 2.875 | 0.854 | [0.562, 1.250] | 0.125 | 1.000 | 4/0/0 |
| all_ok | O semantic vs rank slab | organizational_quality | 4 | 4.083 | 2.500 | 1.583 | [1.167, 1.917] | 0.125 | 1.000 | 4/0/0 |
| all_ok | O semantic vs rank slab | critical_synthesis | 4 | 3.167 | 2.750 | 0.417 | [0.083, 0.833] | 0.250 | 1.000 | 3/1/0 |
| all_ok | O semantic vs rank slab | global_coherence | 4 | 4.083 | 2.917 | 1.167 | [0.667, 1.917] | 0.125 | 1.000 | 4/0/0 |
| all_ok | O semantic vs rank slab | citation_plausibility | 4 | 3.583 | 3.333 | 0.250 | [-0.417, 0.833] | 0.625 | 0.500 | 3/0/1 |
| all_ok | Guarded V vs unguarded stress | overall | 4 | 3.500 | 2.229 | 1.271 | [0.583, 1.771] | 0.125 | 1.000 | 4/0/0 |
| all_ok | Guarded V vs unguarded stress | organizational_quality | 4 | 3.833 | 2.167 | 1.667 | [1.167, 2.167] | 0.125 | 1.000 | 4/0/0 |
| all_ok | Guarded V vs unguarded stress | critical_synthesis | 4 | 2.833 | 2.000 | 0.833 | [0.000, 1.667] | 0.375 | 0.500 | 3/0/1 |
| all_ok | Guarded V vs unguarded stress | global_coherence | 4 | 4.083 | 2.000 | 2.083 | [1.750, 2.500] | 0.125 | 1.000 | 4/0/0 |
| all_ok | Guarded V vs unguarded stress | citation_plausibility | 4 | 3.250 | 2.750 | 0.500 | [-0.667, 1.500] | 0.500 | 0.500 | 3/0/1 |
