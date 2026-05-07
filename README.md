## Hlavné skripty

| Súbor | Popis |
|---|---|
| `get_all_users.py` | Stiahne tréningové dáta z Firebase Storage pre všetkých používateľov. S parametrom `--uid` iba pre konkrétneho používateľa. |
| `fix_data.py` | Opraví formát keystroke CSV súborov |
| `vectors_creation.py` | Vytvorí feature vektory pre všetkých používateľov |
| `prepare_training_data.py` | Pripraví vektory na tréning modelov |
| `run_models_creation_pipeline.py` | Spustí celý tréningový pipeline (stiahnutie dát → tvorba vektorov → príprava vektorov → tréning). S parametrom `--uid` iba pre konkrétneho používateľa. |
| `load_auth_data.py` | Stiahne autentifikačné dáta z Firebase Storage |
| `auth_vector_creation.py` | Vytvorí vektor príznakov pri autentifikácii |
| `auth_server.py` | FastAPI server obsluhujúci autentifikačné požiadavky |
| `AUTHENTICATION.py` | Načíta model, vyhodnotí vektor a vráti rozhodnutie ACCEPT/REJECT |

## Priečinky

### `data/`
Trénovacie a autentifikačné dáta rozdelené podľa fázy spracovania a scenára (common / personal). Každý podpriečinok používateľa je pomenovaný pomocou jeho UID z Firebase.

| Priečinok | Popis |
|---|---|
| `authentication/{uid}/` | Dáta z autentifikačnej požiadavky – surové súbory (keystrokes, senzory) aj vypočítaný feature vektor (`vector_authentication.csv`). |
| `raw_common/{uid}/` | Surové keystroke súbory pre spoločný text po stiahnutí z Firebase, vrátane celých aj čiastkových súborov orezaných na prvých N znakov vstupu (`_10`, `_20`, `_25`, `_50`, `_75`) pripravených na experimenty. |
| `raw_personal/{uid}/` | To isté ako `raw_common`, ale pre vlastný (personal) text. |
| `vectors/` | Čisté vektory bez label stĺpca pre každého používateľa (spoločný text, plný vstup) – **nie sú pripravené na tréning**. |
| `vectors_{N}/` | Čisté vektory bez labelu pre spoločný text orezaný na prvých N znakov. |
| `vectors_personal/` | Čisté vektory bez labelu pre vlastný text (plný vstup). |
| `vectors_personal_{N}/` | Čisté vektory bez labelu pre vlastný text orezaný na prvých N znakov. |
| `training/` | Vektory s label stĺpcom pre každého používateľa (spoločný text, plný vstup) – **pripravené na tréning modelov**. |
| `training_{N}/` | Tréningové súbory pre spoločný text orezaný na prvých N znakov. |
| `training_personal/` | Tréningové súbory s labelmi pre vlastný text (plný vstup). |
| `training_personal_{N}/` | Tréningové súbory pre vlastný text orezaný na prvých N znakov. |

### `RandomForrest/`
Tréning a uložené modely Random Forest pre biometrickú autentifikáciu. Pred spustením trénovania je potrebné prejsť celým pipeline: stiahnutie dát (`get_all_users.py`) → tvorba vektorov (`vectors_creation.py`) → príprava na tréning (`prepare_training_data.py`) → tréning (`RF_train_models_final.py`). Celý pipeline je možné spustiť naraz cez `run_models_creation_pipeline.py`.

| Súbor / priečinok | Popis |
|---|---|
| `RF_train_models_final.py` | Natrénuje Random Forest model pre každého používateľa a uloží ho do `models/` resp. `models_personal/`. |
| `models/` | Natrénované modely (`model_{uid}.pkl`) pre scenár so spoločným textom. |
| `models_personal/` | Natrénované modely (`model_{uid}.pkl`) pre scenár s vlastným textom. |

### `experiments/`
Experimentálne skripty:
- `exp1` – porovnanie RandomForest, SVM a XGBoost
- `exp2` – porovnanie skupín čŕt (keystroke / sensor / combined)
- `exp3` – porovnanie spoločného vs. vlastného textu
- `exp4` – vplyv veľkosti trénovacej množiny na výkon
- `exp5` – vplyv dĺžky vstupného textu na výkon
- `create_truncated_keystrokes.py` – vytvorí orezané keystroke súbory pre exp5
- `visualize_top5_features_SK.py` – načíta produkčné RF modely z `RandomForrest/models/` a `RandomForrest/models_personal/`, vypočíta feature importance (priemer cez všetkých používateľov) a uloží grafy a CSV do `top5_features_visualization/`
- `top5_features_visualization/` – výstupy vizualizácie feature importance z produkčných modelov:
  - `feature_importance_{group}_{dataset}.csv` a `.png` – importance pre každú skupinu čŕt (`keystroke_only`, `sensor_only`, `combined`) a scenár (`general`, `personal`)
  - `feature_importance_all_groups_{dataset}.csv` – všetky skupiny čŕt pre daný scenár v jednom súbore
  - `feature_importance_all_groups_general_vs_personal.csv` – kombinovaný súbor cez oba scenáre
  - `top5_features_combined_GENERAL.png`, `top5_features_combined_PERSONAL.png` – top 5 čŕt ako vertikálne stĺpcové grafy
  - `top5_features_combined_comparison.png` – porovnanie general vs personal vedľa seba
  - `top5_features_combined_comparison.csv` – top 5 čŕt pre oba scenáre
