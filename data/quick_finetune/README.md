# Ready-to-use quick fine-tuning data

These workshop files can be consumed directly by NeMo AutoModel's `ChatDataset`:

- `agent_breaker_train.jsonl`: 512 records (56 analysis, 456 attack)
- `agent_breaker_validation.jsonl`: 64 records (5 analysis, 59 attack)

Each line is a UTF-8 JSON object with one `messages` field containing a non-empty
`user` message followed by a non-empty `assistant` message. There are no duplicate
conversations within a split or shared conversations across splits.

## Provenance

The records are a deterministic, seed-42 stratified sample of the corresponding
train and validation splits in Agent Breaker's
`final_strict_agentslab_family_golden_v16_rebalanced_val_stratified_random`
mixed dataset. Source-only record IDs were removed to match the training schema.
The source is synthetic and classified by Agent Breaker as legacy/unverified, so
this subset is intended only for the authorized defensive-security workshop—not
for production training or benchmarking.

SHA-256:

```text
dfd6ccba5d3d63fc85f17015d318649c5ad8d5afe7e68be16520b8e57dccb6c6  agent_breaker_train.jsonl
4e01e1b283568f8d94f25cb7ed3fc154c82534b3abc4f969af637781d20205e9  agent_breaker_validation.jsonl
```
