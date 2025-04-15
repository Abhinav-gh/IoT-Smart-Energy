# Energy Predictor Preprocessing

This directory contains the preprocessing logic for the ASHRAE Great Energy Predictor III dataset. The preprocessing pipeline allows flexible data reduction modes to speed up experimentation and model prototyping.

## 📜 Script
- `preprocessing.py`: Core script to clean, process, and reduce the dataset.

## 🧪 Usage
Run the script with one of the following options:

```bash
# Full Dataset
python preprocessing.py

# Faster Mode (290 buildings)
python preprocessing.py --faster

# Tiny Mode (10 buildings)
python preprocessing.py --tiny
```

## ⚙️ Modes Explained
| Mode     | Description                                 |
|----------|---------------------------------------------|
| Default  | Uses the full dataset (all buildings/sites) |
| --faster | Keeps 290 buildings for quicker training    |
| --tiny   | Keeps 5 buildings from 2 sites (10 total)   |

## 📉 Tiny Mode Summary (for testing/debugging)
- Reduces the dataset to **5 buildings from site 0** and **5 from site 1**.
- Ideal for local testing, debugging pipelines, and verifying transformations.

### Example Output (Tiny Mode)
```text
INFO: Removed 20110696 rows from train dataset (20110696 / 20216100)
INFO: Removed 41487360 rows from test dataset (41487360 / 41697600)
INFO: Removed 1439 rows from building metadata (1439 / 1449)
INFO: Removed 122226 rows from train weather dataset (122226 / 139773)
INFO: Removed 242436 rows from test weather dataset (242436 / 277243)
```

## 🧼 Preprocessing Includes:
- Timestamp standardization
- Weather cleanup and interpolation
- Feature engineering (lag features, datetime parts)
- Dataset merging
- Memory optimization

## 💾 Outputs
- `processed_train.csv`
- `processed_test.csv`

## 🕒 Sample Timings
| Step                  | Time      |
|-----------------------|-----------|
| Timestamp Processing  | 0.22 sec  |
| Weather Processing    | 96.15 sec |
| Lag Feature Addition  | 0.34 sec  |
| Merging               | 0.16 sec  |
| Feature Engineering   | 0.96 sec  |
| Memory Cleanup        | 0.11 sec  |

---

For more detailed logging and shapes, refer to the preprocessing report output by the script.

