# RH20T GT chunk audit

This test-only tool reuses the frozen iter10000 offline gate, then expands each query into:

```text
current[7]
raw[32,7]
post[32,7]
gt[32,7]
```

`per_axis_horizon.csv` is the lossless comparison table. Each query also owns one `chunks.npz` and `metrics.json` directory. Existing output is never overwritten.

Run on the single-GPU inference host:

```bash
bash tools/rh20t/gt_chunk_audit/run_iter10000.sh
```
