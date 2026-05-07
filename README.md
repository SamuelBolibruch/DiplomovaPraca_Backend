## Hlavné skripty

| Súbor | Popis |
|---|---|
| `get_all_users.py` | Stiahne tréningové dáta z Firebase Storage pre všetkých používateľov. S parametrom `--uid` iba pre konkrétneho používateľa. |
| `fix_data.py` | Opraví formát keystroke CSV súborov |
| `vectors_creation.py` | Vytvorí feature vektory pre všetkých používateľov |
| `prepare_training_data.py` | Pripraví vektory na tréning modelov |
| `run_models_creation_pipeline.py` | Spustí celý tréningový pipeline (stiahnutie dát → vektory → príprava → tréning). S parametrom `--uid` iba pre konkrétneho používateľa. |
| `load_auth_data.py` | Stiahne autentifikačné dáta z Firebase Storage |
| `auth_vector_creation.py` | Vytvorí feature vektor pri autentifikácii |
| `auth_server.py` | FastAPI server obsluhujúci autentifikačné požiadavky |
| `AUTHENTICATION.py` | Načíta model, vyhodnotí vektor a vráti rozhodnutie ACCEPT/REJECT |

## Priečinky

### `data/`
Trénovacie a autentifikačné dáta rozdelené podľa fázy spracovania a scenára (common / personal). Každý podpriečinok používateľa je pomenovaný jeho UID z Firebase.

| Priečinok | Popis |
|---|---|
| `authentication/{uid}/` | Dáta z autentifikačnej požiadavky – surové súbory (keystrokes, senzory) aj vypočítaný feature vektor (`vector_authentication.csv`). |
| `raw_common/{uid}/` | Surové keystroke súbory pre spoločný text po stiahnutí z Firebase, vrátane celých aj čiastkových súborov orezaných na prvých N znakov vstupu (`_10`, `_20`, `_25`, `_50`, `_75`) pripravených na experimenty. |
| `raw_personal/{uid}/` | To isté ako `raw_common`, ale pre osobný (personal) text. |
| `vectors/` | Čisté vektory bez label stĺpca pre každého používateľa (spoločný text, plný vstup) – **nie sú pripravené na tréning**. |
| `vectors_{N}/` | Čisté vektory bez labelu pre spoločný text orezaný na prvých N znakov. |
| `vectors_personal/` | Čisté vektory bez labelu pre osobný text (plný vstup). |
| `vectors_personal_{N}/` | Čisté vektory bez labelu pre osobný text orezaný na prvých N znakov. |
| `training/` | Vektory s label stĺpcom pre každého používateľa (spoločný text, plný vstup) – **pripravené na tréning modelov**. |
| `training_{N}/` | Tréningové súbory pre spoločný text orezaný na prvých N znakov. |
| `training_personal/` | Tréningové súbory s labelmi pre osobný text (plný vstup). |
| `training_personal_{N}/` | Tréningové súbory pre osobný text orezaný na prvých N znakov. |

### `RandomForrest/`
- `RF_train_models_final.py` – tréning Random Forest modelov pre všetkých používateľov

### `experiments/`
Experimentálne skripty:
- `exp1` – porovnanie RandomForest, SVM a XGBoost
- `exp2` – porovnanie skupín čŕt (keystroke / sensor / combined)
- `exp3` – porovnanie shared vs. personal textu
- `exp4` – vplyv veľkosti trénovacej množiny na výkon
- `exp5` – vplyv dĺžky vstupného textu na výkon
- `create_truncated_keystrokes.py` – vytvorí orezané keystroke súbory pre exp5
- `visualize_top5_features_SK.py` – vizualizácia top 5 najdôležitejších čŕt
