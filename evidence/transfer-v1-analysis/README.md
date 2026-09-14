# Audited results

| Branch | Seed | Recall | Transfer | Length 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| examples | base | 14/16 | 89/96 | 24/24 | 21/24 | 24/24 | 20/24 |
| none | base | — | 0/96 | 0/24 | 0/24 | 0/24 | 0/24 |
| rkl_checked | 17 | 3/16 | 3/96 | 1/24 | 1/24 | 1/24 | 0/24 |
| rkl_checked | 29 | 4/16 | 1/96 | 1/24 | 0/24 | 0/24 | 0/24 |
| rkl_onpolicy | 17 | 0/16 | 0/96 | 0/24 | 0/24 | 0/24 | 0/24 |
| rkl_onpolicy | 29 | 3/16 | 2/96 | 1/24 | 1/24 | 0/24 | 0/24 |
| rule | base | — | 65/96 | 17/24 | 15/24 | 17/24 | 16/24 |
| sft | 17 | 16/16 | 24/96 | 14/24 | 5/24 | 2/24 | 3/24 |
| sft | 29 | 16/16 | 32/96 | 12/24 | 7/24 | 7/24 | 6/24 |

| Method | Seed | Training seconds | Teacher seconds | Rollout seconds | Update seconds | Loss tokens |
|---|---:|---:|---:|---:|---:|---:|
| sft | 17 | 23.64 | 0.00 | 0.00 | 23.13 | 1184 |
| rkl_checked | 17 | 81.80 | 58.29 | 0.00 | 22.91 | 1184 |
| rkl_onpolicy | 17 | 131.83 | 57.15 | 51.42 | 22.83 | 1809 |
| sft | 29 | 23.68 | 0.00 | 0.00 | 23.16 | 1184 |
| rkl_checked | 29 | 82.07 | 58.42 | 0.00 | 22.98 | 1184 |
| rkl_onpolicy | 29 | 121.73 | 56.99 | 42.01 | 22.27 | 1282 |

Paired contrasts (percentage points; descriptive identifier-cluster intervals):

- rkl_onpolicy - sft: -28.12; interval -40.62 to -16.67; seeds [-25.0, -31.25]. Positive-advantage rule: False.
- rkl_onpolicy - rkl_checked: -1.04; interval -2.60 to 0.00; seeds [-3.12, 1.04]. Positive-advantage rule: False.
