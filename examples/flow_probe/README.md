# Flow probe faults: survey scripts

Prototypes behind [`prompts/flow_probe_faults.md`](../../prompts/flow_probe_faults.md),
kept to reproduce its numbers. Code reuse should go through
`magnet_scipy.flow_probe`, not through these scripts.

They read pupitre files from `~/github/python_magnetrun/srvdata` (M10 survey)
and `~/LNCMIG-Data/records/srv-data-install/M9` (M9). Each writes its CSV
outputs into the current directory, so run them from a scratch directory.

| script | detector | reads | writes | what |
|---|---|---|---|---|
| `scan.py OUT` | — | srvdata | `OUT` (scan.csv) | zeros and robust flow~rpm residual per group-file |
| `scan2.py OUT` | — | srvdata (2 500 random) | `OUT` (scan2.csv) | static rpm vs lagged rpm vs head predictors |
| `zero2.py` | — | `scan.csv` | `zero2.csv` | zero runs split by head: probe dropout vs pump off |
| `census.py OUT` | v1 `detect.py` | srvdata | `OUT` | fault census, all sites |
| `bench.py` | v1 | `scan2.csv` | `bench.csv` | gap-filling benchmark, 400 group-files |
| `campaign.py` | v2 `detect2.py` | M9_M23072801 provenance + pupitre | `campaign_{runs,events}.csv` | the 24 campaign runs, with `flow_params` |
| `census_m9.py` | v2 | M9 archive | `m9_{runs,events}.csv` | M9 census 2011–2026 |
| `bench_m9.py` | v2 | campaign | `bench_m9.csv` | gap filling: per-file fit vs `flow_params` prior vs head |

`detect.py` (v1) is the first detector, used for the M10 survey numbers.
`detect2.py` (v2) adds the idle-head dropout rule, the flowmeter lag on
the head witness, a fixed τ per magnet, and stale vs stuck by ramp.

Run times on 16 processes: `scan.py` ~1.5 min, `census.py` ~1 min,
`census_m9.py` ~3.5 min, the rest under a minute.
