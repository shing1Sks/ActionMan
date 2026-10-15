# v3 data audit (S0)

Shares are over gold argument values. 'copyable' = the value text occurs in the query, case-insensitively, on token boundaries.

## train_v2.jsonl

| source | args | str copyable | num copyable | bool/enum (options) | gold in options | list args | list items copyable | item >8 tok | list >7 | value >32 tok |
|---|---|---|---|---|---|---|---|---|---|---|
| hermes | 5201 | 80.5% of 3700 | 83.1% of 455 | 11.6% | 81.0% | 14.0% | 70.7% | 20.8% | 9 | 7.2% |
| toolace | 29526 | 79.1% of 19705 | 50.1% of 6043 | 3.9% | 100.0% | 6.2% | 25.7% | 67.1% | 16 | 5.4% |
| xlam | 162625 | 71.8% of 93760 | 80.4% of 55290 | 1.3% | 98.3% | 6.7% | 86.0% | 2.8% | 1390 | 0.3% |

## test.jsonl

| source | args | str copyable | num copyable | bool/enum (options) | gold in options | list args | list items copyable | item >8 tok | list >7 | value >32 tok |
|---|---|---|---|---|---|---|---|---|---|---|
| xlam | 2719 | 74.0% of 1605 | 79.8% of 875 | 1.5% | 95.0% | 7.1% | 80.1% | 2.6% | 18 | 0.2% |

## bfcl.jsonl

| source | args | str copyable | num copyable | bool/enum (options) | gold in options | list args | list items copyable | item >8 tok | list >7 | value >32 tok |
|---|---|---|---|---|---|---|---|---|---|---|
| bfcl_multiple | 515 | 80.3% of 254 | 82.1% of 195 | 7.6% | 100.0% | 8.3% | 85.6% | 2.2% | 0 | 0.2% |
| bfcl_parallel | 1440 | 91.5% of 610 | 86.2% of 682 | 7.6% | 100.0% | 6.2% | 94.5% | 0.0% | 6 | 0.1% |
| bfcl_parallel_multiple | 1562 | 90.9% of 843 | 69.4% of 576 | 5.1% | 100.0% | 6.3% | 94.6% | 0.0% | 1 | 0.1% |
| bfcl_simple | 1059 | 81.3% of 540 | 77.1% of 397 | 7.8% | 100.0% | 7.3% | 74.9% | 0.9% | 0 | 0.2% |
