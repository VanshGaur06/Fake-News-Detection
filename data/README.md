# LIAR Dataset

This folder contains the official LIAR benchmark dataset splits:
- `train.tsv`: 10,240 labeled training statements
- `valid.tsv`: 1,284 validation statements
- `test.tsv`: 1,283 test statements

### Columns (14 tab-separated values)
1. `id`: Statement ID (e.g., `2635.json`)
2. `label`: 6-way verdict (`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`)
3. `statement`: The news claim or political quote
4. `subject`: Topic (e.g., economy, healthcare, taxes)
5. `speaker`: Person who made the statement
6. `speaker_job`: Job title of the speaker
7. `state_info`: State or region
8. `party`: Political affiliation (e.g., Democrat, Republican)
9. `barely_true_count`: Speaker's historical barely-true claims
10. `false_count`: Speaker's historical false claims
11. `half_true_count`: Speaker's historical half-true claims
12. `mostly_true_count`: Speaker's historical mostly-true claims
13. `pants_on_fire_count`: Speaker's historical pants-on-fire claims
14. `context`: Context / venue (e.g., debate, interview, press release)

### Binary Mapping Used in TruthLens
- **FAKE**: `pants-fire`, `false`, `barely-true`
- **REAL**: `half-true`, `mostly-true`, `true`
