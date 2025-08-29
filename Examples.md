# Magnet-Scipy

* voltage mode

* using `main`:

```bash
python -m magnet_scipy.main \
  --experimental_csv examples/M19/M19_Overview_240208-0941_current.csv \
  --voltage_csv examples/M19/M19_Overview_240208-0941-Tensions_Aimant_ALL_externes.csv \
  --resistance_csv examples/M19/Rtot_M9Bitters_18MW.csv \
  --inductance 0.01254 \
  --temperature 12.2 \
  --value_start 200 \
  --time_start 0 --time_end 6000  --time_step 0.01
```

* using `couple_main`:

```bash
python -m magnet_scipy.coupled_main \
  --wd examples/M19/  \
  --config-file M9Bitters_18MW.json \
  --value_start 200  \
  --time_end 6000 --time_step 0.01 \
  --experimental_csv M19_Overview_240208-0941_current.csv
```
with

```json
{
  "name": "M9_xxx",
  "circuits": [
    {
      "circuit_id": "M9Bitters_18MW",
      "temperature": 12.2,
      "inductance": 0.01254,
      "resistance_csv": "Rtot_M9Bitters_18MW.csv",
      "voltage_csv": "M19_Overview_240208-0941-Tensions_Aimant_ALL_externes.csv"
    }
  ],
  "mutual_inductances": []
}
```

* pid mode - running pid each time_step

```bash
python -m magnet_scipy.main \
    --wd examples/M19/ \
    --experimental_csv M19_Overview_240208-0941_voltage.csv \
    --reference_csv M19_Overview_240208-0941-Courants_Alimentations_Référence_GR2.csv \
    --resistance_csv Rtot_M9Bitters_18MW.csv \
    --inductance 0.01254 \
    --temperature 12.2 \
    --value_start 200 \
    --time_start 0 --time_end 6000  --time_step 0.01 \
  --custom_pid \
  --kp_low 5 --ki_low 0.2 --kd_low 0 \
  --kp_medium 12 --ki_medium 1 --kd_medium 0 \
  --kp_high 12 --ki_high 1 --kd_high 0 \
  --low_threshold 60 \
  --high_threshold 800
```



PID params:
* Kp=5, Ki=0.2, Kd=0 for low current region (60A)
* Kp=12, Ki=1, Kd=0 for medium current region (800A)
* Kp=12, Ki=1, Kd=0 for high current region 

```bash
python -m python_magnetrun.analysis-refactor pigbrotherdata/Fichiers_Data/M9/Overview/M9_Overview_250303-*.tdms  --key Référence_GR1 --show --synchronize
```

```bash
python tests/test-signature.py pigbrotherdata/Fichiers_Data/M19/Overview/M19_Overview_240208-0941.tdms --key Courants_Alimentations/Référence_GR1 --threshold 0.5
```


```bash
python -m tests.test-signature srvdata/M10_2025.01.27---15:39:29.txt --window=10 --threshold 1.e-2
```

```bash
python3 -m python_magnetrun.python_magnetrun srvdata/M9_2019.02.14---23\:00\:38.txt info --list
```

```bash
python -m python_magnetrun.python_magnetrun fichier select --output_key key1 key2 key3 ... --->  cree un fichier fichier_key1_key2_key3_vs_Time.csv
```

```bash
python -m python_magnetrun.python_magnetrun ~/M19_Overview_240208-0941.tdms \
    plot --vs_time Courants_Alimentations/Courant_GR1 Courants_Alimentations/Courant_GR2
```

```bash
python -m python_magnetrun.python_magnetrun ~/M9_Overview_240208-0941.tdms ~/M9_2024.05.09---16_34_03.txt \
    plot --vs_time Courants_Alimentations/Courant_GR1 --vs_time IH
```

Tensions_Aimant/ALL_externes / GR2
Tensions_Aimant/ALL_externes_2 / GR1 (i0=1000A)

# Test case

"Kp_low": 20.0, "Ki_low": 15.0, "Kd_low": 0.1
"Kp_medium": 12.0, "Ki_medium": 8.0, "Kd_medium": 0.05
"Kp_high": 8.0, "Ki_high": 5.0, "Kd_high": 0.02

temperature: 45 °C
R=2.0
L=0.1

in examples data directory:
resistance_csv = test_variable_resistance()
reference_csv = create_sample_reference_csv

```bash
python -m magnet_diffrax.main \
    --reference_csv reference_current.csv \
    --resistance_csv test_resistance.csv \
    --inductance 0.1 \
    --temperature 45 \
    --value_start 0 \
    --time_start 0 \
    --time_end 8 \
  --custom_pid \
  --kp_low 20 --ki_low 15 --kd_low 0.1 \
  --kp_medium 12 --ki_medium 8 --kd_medium 0.05 \
  --kp_high 8 --ki_high 5 --kd_high 0.02 \
  --low_threshold 60 \
  --high_threshold 800 \
  --show-plots
```

# M19 test case

```
"M9": {"Référence_GR1": ["UH"], "Référence_GR2": ["UB", "Ucoil15", "Ucoil16"]},
"M10": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
```

## config Alim

see python_magnetrun/todos directory:
* Groupe1: A1_version_config_24.xml,  A2_version_config_24.xml  
* Groupe2: A3_version_config_45.xml,  A4_version_config_44.xml

Master: A1, A3
Slave: A2, A4

To read xml data: `python_magnetrun/todos/convertxml.py`

PID params:
* Groupe2: 
Current PID M9: Numero Seuil: 1, Ki: 0.2, Kp: 5, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M9: Numero Seuil: 2, Ki: 1, Kp: 12, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M9: Numero Seuil: 3, Ki: 1, Kp: 12, Kd: -1, Rapport Entre Boucle: 0.75

* Groupe1:
Current PID M10: Numero Seuil: 1, Ki: 1, Kp: 10, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M10: Numero Seuil: 2, Ki: 1, Kp: 10, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M10: Numero Seuil: 3, Ki: 1, Kp: 10, Kd: -1, Rapport Entre Boucle: 0.75


## pupitre

* files: '2024.02.08 - 09_07_57.txt' (M10), '2024.02.08 - 09_07_52.txt' '2024.02.08 - 12_54_21.txt' (M9)

```
"M9": {"Référence_GR1": ["UH"], "Référence_GR2": ["UB", "Ucoil15", "Ucoil16"]},
"M10": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
```

```
"M9": {"Tin2": "T_B", "Tin1": "T_H"},
"M10": {"Tin2": "T_H", "Tin1": "T_B"},
```

M9: Tin1= 11.6, Tin2= 12.2
M10:  Tin1= 11.6, Tin2= 12.2

## pigbrother

* files: `M19_Overview_240208-0941.tdms`, `M19_Archive_240208-0941.tdms`

```
Tensions_Aimant         M9Externe1            16174            1  2024-02-08T08:41:18.592752             0.5
Tensions_Aimant         M9Externe2            16174            1  2024-02-08T08:41:18.592752             0.5
Tensions_Aimant         ALL_externes          16174            1  2024-02-08T08:41:18.592752             0.5
```

```
Tensions_Aimant         M10Externe1           16174            1  2024-02-08T08:41:18.592752             0.5
Tensions_Aimant         M10Externe2           16174            1  2024-02-08T08:41:18.592752             0.5
Tensions_Aimant         ALL_externes_2        16174            1  2024-02-08T08:41:18.592752             0.5
```

# todos

* [x] turn into package
* [x] add pytest
* [] add docs with sphinx
* [] add benchmarks

* read with resistance with a given key
* read temperature from csv (extracted from pupitre data)
* can load inductance from csv
* use data from signature instead of plain "csv" for inputs
* intermediate points for voltage

* coupled case
  * how to handle mutual inductance? csv file? (what if we load detailled inductance matrix - mean with inductance/mutual for each "part")
  * plotting: plots per circuit
  * analytics: .... per circuit
  * rework intermediate points for voltage
  * what to do with screens? must be parallel circuits without PID nor applied voltage
  * how to handle current source for supra? thevenin <-> norton
  * how to handle protection circuit in supra

PS: check inductance/mutual calculation in python3-magnettools (see github/magnettools/Python/self_mutual.p) and in python3_magnetworkflow (Axi and 3D)