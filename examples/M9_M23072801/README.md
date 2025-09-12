# config Alim

PID params:
* Groupe2:
Current PID M9: Numero Seuil: 1, Ki: 2, Kp: 4, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M9: Numero Seuil: 2, Ki: 2, Kp: 4, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M9: Numero Seuil: 3, Ki: 2, Kp: 6, Kd: -1, Rapport Entre Boucle: 0.75

* Groupe1:
Current PID M9: Numero Seuil: 1, Ki: 0.2, Kp: 5, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M9: Numero Seuil: 2, Ki: 1, Kp: 12, Kd: -1, Rapport Entre Boucle: 0.75
Current PID M9: Numero Seuil: 3, Ki: 1, Kp: 12, Kd: -1, Rapport Entre Boucle: 0.75

# config site M9

* M9Bitters 18 MW
* M23072801

## pupitre

* files:

```
M9_2023.02.08---15:46:27.txt
M9_2023.09.11---14:21:59.txt
M9_2023.09.13---22:05:38.txt
M9_2023.09.14---11:48:26.txt
M9_2023.09.14---21:43:07.txt
M9_2023.09.15---20:07:01.txt
M9_2023.09.16---17:55:22.txt
```

```
"M9": {"Référence_GR1": ["UH"], "Référence_GR2": ["UB", "Ucoil15", "Ucoil16"]},
```

```
"M9": {"Tin2": "T_B", "Tin1": "T_H"},
```

```
M9_2023.09.11---14:21:59.txt : Tin2 range(16.5, 22), Tin1 range(16.5, 22.5)
M9_2023.09.13---22:05:38.txt : Tin2 range(,18.6), Tin1 range(,18.1)
M9_2023.09.14---11:48:26.txt : Tin2 range(,22,2), Tin1 range(, 21.6)
M9_2023.09.14---21:43:07.txt : Tin2 range(, 21.9), Tin1 range(, 21.4)
M9_2023.09.15---20:07:01.txt : Tin2 range(, 21.8), Tin1 range(, 21.3)
M9_2023.09.16---17:55:22.txt : Tin2 range(, 19), Tin1 range(, 18.5) ** 
```


## pigbrother

```
Tensions_Aimant         ALL_internes          16174            1  2024-02-08T08:41:18.592752             0.5
Tensions_Aimant         ALL_externes          16174            1  2024-02-08T08:41:18.592752             0.5
```

```
Courants_Alimentations  Référence_GR1
Courants_Alimentations  Référence_GR2
Courants_Alimentations  Courant_GR1
Courants_Alimentations  Courant_GR2
```

* files:

```
M9_Overview_230911-1041.tdms
M9_Overview_230911-1051.tdms
M9_Overview_230911-1143.tdms
M9_Overview_230911-1334.tdms
M9_Overview_230911-1343.tdms
M9_Overview_230913-2205.tdms
M9_Overview_230914-1148.tdms
M9_Overview_230914-2143.tdms
M9_Overview_230915-1633.tdms
M9_Overview_230915-2006.tdms
M9_Overview_230916-1755.tdms
```

```
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230911-1041.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230911-1051.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230911-1143.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230911-1334.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230911-1343.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230913-2205.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230914-0305.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230914-1148.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230914-2143.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230915-1633.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230915-2006.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230916-0106.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230916-0605.tdms
/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data/M9/Fichiers_Archive/M9_Archive_230916-1755.tdms
```

NB: to extract data as csv

```bash
python -m python_magnetrun.python_magnetrun M9_2023.09.16---17\:55\:22.txt select --output_key UH UB TinH TinB FlowH FlowB teb debitbrut
python -m python_magnetrun.python_magnetrun M9_2023.09.16---17\:55\:22.txt select --output_key TinH
python -m python_magnetrun.python_magnetrun M9_2023.09.16---17\:55\:22.txt select --output_key TinB
python -m python_magnetrun.python_magnetrun M9_M23072801/M9_Overview_230916-1755.tdms select --output_key Courants_Alimentations/Courant_GR1 Courants_Alimentations/Courant_GR2 
```

For data comming from `pupitre` format, we need to smooth then before export and consider a shift to account for the difference en time between `pupitre` and `pigbrother`.

You can also extract input data - aka voltage or reference currents - using signature (see magnetrun)
There is a problem in signature connected with eco mode (U sequence is not necessary linear - introduce several U instead??)

## inital conditions

"i0": "IH": -6.32 A, "IB": 4.04 A for M9_2023.09.16---17:55:22.txt, M9_Overview_230916-1755.tdms
