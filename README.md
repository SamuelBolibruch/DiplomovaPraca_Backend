## Hlavné skripty

| Súbor | Popis |
|---|---|
| `get_all_users.py` | Stiahne súbory z Firebase Firestore potrebné na vytvorenie vektorov |
| `fix_data.py` | Upraví pôvodné CSV na správny formát |
| `vectors_creation.py` | Vytvorí vektory pre všetkých používateľov |
| `prepare_training_data.py` | Pripraví vektory používateľov pre tréning modelov |
| `run_models_creation_pipeline.py` | Spustí celý pipeline: stiahnutie dát → tvorba vektorov → príprava tréningových dát → tréning modelov |
| `load_auth_data.py` | Stiahne súbory potrebné na autentifikáciu |
| `auth_vector_creation.py` | Vytvorí vektor pri autentifikácii |
| `auth_server.py` | Spustí FastAPI server obsluhujúci autentifikačné požiadavky |
| `AUTHENTIFICATION.py` | Vykoná autentifikáciu pre konkrétneho používateľa (načíta model, predikuje) |
| `AUTHENTIFICATION_pipeline.py` | Spustí celý autentifikačný pipeline pre zadané `--uid` |
| `create_truncated_keystrokes.py` | Vytvorí orezané verzie keystroke súborov (N = 10, 20, 50, 75 znakov) |
| `find_longest_personal_sentence.py` | Nájde najdlhšiu osobnú vetu v datasete |
| `serviceAccountKey.json` | Kľúč pre prístup k Firebase / Firestore |

---

## Priečinky

### `data/`
Obsahuje trénovacie a autentifikačné dáta – vektory rozdelené podľa scenárov (general / personal) a veľkostí trénovacej množiny (10, 20, 25, 50, 75 %).

### `randomforrest/`
- `RF_train_models_final.py` – finálny tréning modelov pre všetkých používateľov

### `experiments/`
Experimentálne skripty porovnávajúce modely a parametre:
- `exp1` – porovnanie RandomForest, SVM a XGBoost
- `exp2` – porovnanie skupín čŕt (keystroke / sensor / combined)
- `exp3` – porovnanie shared vs. personal textu
- `exp4` – vplyv veľkosti trénovacej množiny na výkon
- `exp5` – vplyv dĺžky vstupného textu na výkon
- `visualize_top5_features*.py` – vizualizácia top 5 najdôležitejších čŕt
