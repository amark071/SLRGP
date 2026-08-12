# S4a main8 deterministic audit and descriptive statistics

## Run identity

- Output directory (this package): `data/exp3_interface_substitution/s4a_main8_20260711_173157`
- Total topic-arm generations: 40 = 8 topics x 5 arms
- Successful generations: 40/40
- Deterministic audit issues: 0
- Final reported LLM usage: 313 calls; 270,415 prompt tokens; 75,532 completion tokens

## Deterministic audit

- All 40 runs have survey, metadata, trace, references, evidence package, and leaf-evidence provenance files.
- No duplicate Markdown headings were detected.
- All leaf writes have candidate provenance.
- All cited IDs in generated text are contained in leaf-level evidence provenance.
- No failed or incomplete arm remains after resume.

## Arm-level descriptive means

| arm | words | refs | elapsed_s | LLM_calls | solve_events | descend_events | leaf_writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| intact | 968.12 | 10.50 | 87.40 | 11.88 | 8.00 | 2.75 | 2.38 |
| l_default | 1162.12 | 14.25 | 109.64 | 13.38 | 8.12 | 3.50 | 2.75 |
| r_pass | 1156.25 | 12.75 | 110.58 | 13.62 | 8.25 | 3.50 | 2.75 |
| o_rank_slab | 828.75 | 11.50 | 39.35 | 3.00 | 3.00 | 2.00 | 2.00 |
| no_recursion | 848.75 | 13.62 | 76.09 | 8.62 | 4.50 | 2.00 | 2.00 |

## Paired mean difference vs intact

| ablation_arm | diff_words | diff_refs | diff_elapsed_s | diff_LLM_calls | diff_solve_events | diff_descend_events | diff_leaf_writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| l_default | 194.00 | 3.75 | 22.24 | 1.50 | 0.12 | 0.75 | 0.38 |
| r_pass | 188.12 | 2.25 | 23.17 | 1.75 | 0.25 | 0.75 | 0.38 |
| o_rank_slab | -139.38 | 1.00 | -48.05 | -8.88 | -5.00 | -0.75 | -0.38 |
| no_recursion | -119.38 | 3.12 | -11.31 | -3.25 | -3.50 | -0.75 | -0.38 |

## Immediate interpretation

- The run validates the S4a generation substrate: all arms completed under shared topic set, shared backbone, fixed budget, and preserved provenance.
- Deterministic metadata alone is not a final quality judgment. Quality claims still require Instrument J/blind judging or a registered deterministic quality proxy.
- Operationally, l_default and r_pass require more time/calls than intact on average, while o_rank_slab is much cheaper because it removes semantic organization calls.
- Therefore, the current evidence supports an interface/trace-level necessity check, but not yet a standalone claim that intact has higher textual quality than every ablation.