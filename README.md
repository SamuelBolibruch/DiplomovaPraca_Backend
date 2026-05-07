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

| Súbor | Popis | Výsledky |
|---|---|---|
| `exp1_model_comparison.py` | Porovnáva Random Forest, SVM a XGBoost na oboch datasetoch (general, personal). Pre každý model vypočíta AAR, FAR, FRR, EER. | `results/exp1_model_comparison/` |
| `exp2_RF_feature_group_comparison.py` | Porovnáva skupiny čŕt (`keystroke_only`, `sensor_only`, `combined`) pomocou Random Forest na oboch datasetoch. | `results/exp2_RF_feature_group_comparison/` |
| `exp2_SVM_feature_group_comparison.py` | To isté ako exp2 RF, ale s modelom SVM. | `results/exp2_SVM_feature_group_comparison/` |
| `exp2_XGB_feature_group_comparison.py` | To isté ako exp2 RF, ale s modelom XGBoost. | `results/exp2_XGB_feature_group_comparison/` |
| `exp3_RF_text_scenario_comparison.py` | Porovnáva scenáre spoločného vs. vlastného textu pomocou Random Forest. | `results/exp3_RF_text_scenario_comparison/` |
| `exp4_general_training_size.py` | Skúma vplyv veľkosti trénovacej množiny na výkon RF modelu (general aj personal dataset). | `results/exp4_training_size_analysis/` |
| `exp5_RF_text_length_comparison.py` | Porovnáva oba datasety pri rôznych dĺžkach vstupného textu (10, 20, 25, 50, 75, plný). | `results/exp5_RF_text_length_analysis/` |
| `exp5_RF_text_length_general.py` | Vplyv dĺžky vstupného textu na výkon RF – iba general dataset. | `results/exp5_RF_text_length_analysis/general/` |
| `exp5_RF_text_length_personal.py` | Vplyv dĺžky vstupného textu na výkon RF – iba personal dataset. | `results/exp5_RF_text_length_personal/` |
| `create_truncated_keystrokes.py` | Vytvára orezané keystroke CSV súbory (prvých N znakov) – potrebné pred spustením exp5. | `data/raw_common/`, `data/raw_personal/` |
| `visualize_top5_features_SK.py` | Načíta produkčné RF modely, vypočíta priemernú feature importance cez všetkých používateľov a uloží grafy a CSV. | `top5_features_visualization/` |
