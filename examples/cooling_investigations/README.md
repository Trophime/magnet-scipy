# Cooling investigations

Studies of the M9/M10 cooling and power data behind the defaults of
[`../energy_balance.py`](../energy_balance.py) and the cooling-model plan
([`prompts/cooling_model.md`](../../prompts/cooling_model.md)). They read the
pupitre (`srv-data-install/`) and pigbrother (`pbsurv/`) records under
`~/LNCMIG-Data/records` (`--records` to change it).

## Power-supply cooling flow

The power-supply cooling circuit (auxiliary pumps PA1–PA3, two parallel
circuits A1/A2 and A3/A4, one outlet sensor `TAlimout`) has no flowmeter. Its
flow is inferred from the energy balance of each long plateau:

```
F_alim_implied = P_alim / (ρcp·(TAlimout − Tin)),   P_alim = Ptot − ΣPuissance_A − P_pumps
```

`P_alim` is the transformer + converter loss (the circuit also cools the
transformers' water/oil exchangers; `Ptot` includes them and the two magnet
pumps).

### Pipeline

```bash
D=examples/cooling_investigations
R=~/LNCMIG-Data/records/srv-data-install
# 1. per file: is Ptot recorded? (rule Pmagnet * (Ptot == 0), as in pupitre_checks.json)
python $D/scan_ptot_rule.py ptot_rule_scan.csv $R/M9/*.txt $R/M10/*.txt
# 2. steady plateaus of the files where Ptot is recorded (pupitre only)
python $D/plateaus.py ptot_rule_scan.csv plateaus.csv
# 3. implied flow on plateaus >= 30 min above 5 MW, with pigbrother data
python $D/alim_flow_study.py plateaus.csv alim_flow_study.csv
```

Steps 1 and 2 take about 6 min each on 22 cores (11 753 files); step 3 about
as long. Their intermediate outputs (`ptot_rule_scan.csv`, `plateaus.csv`)
are not kept here. Needs `nptdms`, `pandas`, `python_magnetrun` (rule
statistics, dated housing wiring).

### Result: [`alim_flow_study.csv`](alim_flow_study.csv)

One row per plateau (538 plateaus, 294 runs, 2021-11 to 2026-10, the period
with pigbrother Overview files): `housing`, `file`, `start` (UTC), `dur_min`,
`lag_s` (pupitre/pigbrother offset), powers [MW] `P_mag`, `P_pup`, `Ptot`,
`P_busbar`, `P_pump_el`, `P_alim`, temperatures [°C] `Tin`, `TAlim`,
`F_alim_implied` [l/s], `Rpm1`, `Rpm2` [rpm], `Flow1`, `Flow2` [l/s].

Plateaus with `TAlim − Tin` > 3 K (531):

| | M9 | M10 |
|---|---|---|
| Median implied flow | 28.1 l/s (IQR 26.4–33.4) | 30.4 l/s (IQR 20.2–34.3) |
| Scatter within a run | 1.4 l/s | 1.5 l/s |
| Scatter of run medians | 5.1 l/s | 8.9 l/s |
| Levels | mostly 26–28, sometimes 34–38 l/s | 28–36, sometimes 14–20 l/s |

The levels hold for days to weeks, then switch, and in the low M10 periods
`TAlimout − Tin` doubles at the same `P_alim`: the flow really changes,
most likely with the auxiliary-pump configuration. Caveats: the flow is
inferred, not measured; M10 uses the M9 pump coefficient (0.023 MW per pump
per (1000 rpm)³), which shifts its absolute level but not its jumps.

## Related evidence (not scripted here)

- `Ptot` includes the magnet pumps: at constant current, M9 run
  2026.09.04 13:44 had pump 2 moved 2247 → 2780 → 1535 rpm and
  `Ptot − Pmagnet` moved by +0.25 / −0.16 MW (rpm³ law: +0.27 / −0.21 MW).
  This also gives the pump coefficient.
- The `Ptot_not_recorded` rule threshold (1.0 MW, `max_abs`) was chosen from
  the output of `scan_ptot_rule.py`; see
  `hifimagnet-projects/magnetdb.json/pupitre_checks.json`.
