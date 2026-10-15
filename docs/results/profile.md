# Data profile

### xLAM train sample (n=8000)

- Context tokens (compact signatures): p50 186, p90 369, p99 581, max 893
- Same context as OpenAI JSON schema: mean 407 vs compact 209 tokens (2.0x)
- Argument value tokens: p50 2, p90 7, p99 21, max 245
- Examples with > 4 calls: 37 (0.5%)
- Skipped: context too long 0 (0.0%), slot block too long 0

| bucket | share of (call, param) items |
|---|---|
| absent | 17.5% |
| <=4 | 64.7% |
| <=8 | 10.3% |
| <=16 | 5.1% |
| <=32 | 2.1% |
| overflow >32 | 0.3% |

### BFCL simple (n=400)

- Context tokens (compact signatures): p50 92, p90 119, p99 158, max 162
- Same context as OpenAI JSON schema: mean 162 vs compact 94 tokens (1.7x)
- Argument value tokens: p50 2, p90 6, p99 17, max 65
- Examples with > 4 calls: 0 (0.0%)
- Skipped: context too long 1 (0.2%), slot block too long 0

| bucket | share of (call, param) items |
|---|---|
| absent | 8.4% |
| <=4 | 79.7% |
| <=8 | 6.9% |
| <=16 | 4.1% |
| <=32 | 0.8% |
| overflow >32 | 0.2% |

### BFCL multiple (n=200)

- Context tokens (compact signatures): p50 191, p90 293, p99 340, max 373
- Same context as OpenAI JSON schema: mean 397 vs compact 205 tokens (1.9x)
- Argument value tokens: p50 2, p90 6, p99 13, max 40
- Examples with > 4 calls: 0 (0.0%)
- Skipped: context too long 0 (0.0%), slot block too long 0

| bucket | share of (call, param) items |
|---|---|
| absent | 9.3% |
| <=4 | 78.0% |
| <=8 | 8.6% |
| <=16 | 3.7% |
| <=32 | 0.2% |
| overflow >32 | 0.2% |

### BFCL parallel (n=200)

- Context tokens (compact signatures): p50 121, p90 190, p99 240, max 259
- Same context as OpenAI JSON schema: mean 198 vs compact 131 tokens (1.5x)
- Argument value tokens: p50 1, p90 5, p99 16, max 41
- Examples with > 4 calls: 3 (1.5%)
- Skipped: context too long 0 (0.0%), slot block too long 0

| bucket | share of (call, param) items |
|---|---|
| absent | 2.8% |
| <=4 | 87.1% |
| <=8 | 5.9% |
| <=16 | 3.4% |
| <=32 | 0.7% |
| overflow >32 | 0.1% |

### BFCL parallel_multiple (n=200)

- Context tokens (compact signatures): p50 251, p90 398, p99 511, max 522
- Same context as OpenAI JSON schema: mean 429 vs compact 257 tokens (1.7x)
- Argument value tokens: p50 2, p90 5, p99 15, max 94
- Examples with > 4 calls: 1 (0.5%)
- Skipped: context too long 0 (0.0%), slot block too long 0

| bucket | share of (call, param) items |
|---|---|
| absent | 2.8% |
| <=4 | 86.0% |
| <=8 | 6.7% |
| <=16 | 3.7% |
| <=32 | 0.6% |
| overflow >32 | 0.1% |

### BFCL irrelevance (n=240)

- Context tokens (compact signatures): p50 86, p90 103, p99 128, max 156
- Same context as OpenAI JSON schema: mean 154 vs compact 86 tokens (1.8x)
- Argument value tokens: p50 0, p90 0, p99 0, max 0
- Examples with > 4 calls: 0 (0.0%)
- Skipped: context too long 0 (0.0%), slot block too long 0

| bucket | share of (call, param) items |
|---|---|
| absent | 0.0% |
| <=4 | 0.0% |
| <=8 | 0.0% |
| <=16 | 0.0% |
| <=32 | 0.0% |
| overflow >32 | 0.0% |
