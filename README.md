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

---

## Inštalácia a spustenie

### Požiadavky

- **Python 3.13** (projekt bol vyvíjaný a testovaný na tejto verzii)

### 1. Klonovanie repozitára

```bash
git clone <url-repozitara>
cd DiplomovaPraca_Backend
```

### 2. Vytvorenie virtuálneho prostredia

```bash
python3.13 -m venv venv
```

### 3. Aktivácia virtuálneho prostredia

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Po aktivácii by mal byť prefix `(venv)` viditeľný v termináli.

### 4. Inštalácia závislostí

```bash
pip install -r requirements.txt
```

### 5. Firebase Service Account Key

Súbor `serviceAccountKey.json` je potrebný pre skripty, ktoré komunikujú priamo s Firebase Storage a Firestore – konkrétne:

- `get_all_users.py` – stiahne tréningové dáta zo **Storage** pre všetkých (alebo konkrétneho) používateľa
- `load_auth_data.py` – stiahne autentifikačné dáta zo **Storage** pri spracovaní autentifikačnej požiadavky

Bez tohto súboru nie je možné spustiť sťahovanie dát z Firebase. Ostatné skripty (tvorba vektorov, tréning modelov, experimenty, autentifikačný server) tento súbor **nepotrebujú** – pracujú len s lokálne uloženými dátami.

**Ako získať `serviceAccountKey.json`:**

1. V [Firebase Console](https://console.firebase.google.com) otvor svoj projekt.
2. Prejdi do **Project settings → Service accounts**.
3. Klikni na **Generate new private key** a stiahni JSON súbor.
4. Premenuj ho na `serviceAccountKey.json` a umiestni ho do koreňového priečinka projektu (vedľa `get_all_users.py`).

> Tento súbor obsahuje citlivé prihlasovacie údaje – **nikdy ho nezdieľaj ani nepridávaj do gitu.**

### 6. Autentifikačný server

Server beží lokálne na porte `8000` a vystavuje dva endpointy:

| Endpoint | Metóda | Popis |
|---|---|---|
| `/register` | POST | Spustí tréningový pipeline pre daného používateľa (`uid`) |
| `/authenticate` | POST | Stiahne autentifikačné dáta, vytvorí vektor a vráti rozhodnutie ACCEPT/REJECT |

```bash
uvicorn auth_server:app --reload
```

Po spustení je server dostupný na `http://localhost:8000`. Dokumentácia endpointov je automaticky dostupná na `http://localhost:8000/docs`.

**Sprístupnenie servera na internete (napr. pre mobilnú aplikáciu):**

Lokálny server je možné vystaviť navonok napríklad cez [ngrok](https://ngrok.com):

```bash
ngrok http 8000
```

Ngrok vygeneruje verejnú URL (napr. `https://xxxx.ngrok-free.app`), ktorú je možné použiť ako adresu servera v mobilnej aplikácii.

---

### 7. Spustenie skriptov

Po aktivácii prostredia a inštalácii závislostí je možné spúšťať jednotlivé skripty priamo z koreňového priečinka projektu.

> **Poznámka k príkazu `python`:** Na macOS a Linuxe môže byť potrebné použiť `python3` namiesto `python`.

```bash
# Stiahnutie tréningových dát z Firebase (vyžaduje serviceAccountKey.json)
python get_all_users.py

# Stiahnutie iba pre konkrétneho používateľa
python get_all_users.py --uid <uid>

# Celý pipeline: stiahnutie → vektory → príprava → tréning
python run_models_creation_pipeline.py

# Spustenie autentifikačného FastAPI servera
uvicorn auth_server:app --reload
```
