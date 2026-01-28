import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# we maken een pad aan naar de map waar alle data in komt
# Path is handig omdat het netjes werkt op elk besturingssysteem
DATA_DIR = Path("data")

# binnen de data-map maken we een submap voor train/val/test splits
SPLIT_DIR = DATA_DIR / "splits"

# zorg dat de mappen bestaan
# exist_ok=True betekent: geen fout als ze al bestaan
DATA_DIR.mkdir(exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# we maken 100 nep-patiënt-ID's
# TCGA-MOCK-0000 t/m TCGA-MOCK-0099
pids = [f"TCGA-MOCK-{i:04d}" for i in range(100)]

# we definiëren 32 verschillende kankertypes
CLASSES = [f"CANCER_{i}" for i in range(32)]

# we splitsen de patiënt-ID's op in train, validatie en test
# eerste 70 voor training
json.dump(pids[:70], open(SPLIT_DIR / "train.json", "w"))
# volgende 15 voor validatie
json.dump(pids[70:85], open(SPLIT_DIR / "val.json", "w"))
# laatste 15 voor test
json.dump(pids[85:], open(SPLIT_DIR / "test.json", "w"))

# we maken een mapping van kankertypes naar getallen
# dit is nodig omdat modellen met nummers werken, niet met tekst
# bijv. "CANCER_0" -> 0, "CANCER_1" -> 1, etc.
label_map = {c: i for i, c in enumerate(CLASSES)}

# sla deze mapping op als json-bestand
json.dump(label_map, open(DATA_DIR / "label_mapping.json", "w"))

# we maken een lijst met labels per patiënt
# elke patiënt krijgt willekeurig één kankertype toegewezen
df_data = [{"patient_id": p, "cancer_type": np.random.choice(CLASSES)} for p in pids]

# zet dit om in een tabel en sla het op als CSV
# dit lijkt op echte label-bestanden die je in ML-projecten ziet
pd.DataFrame(df_data).to_csv(DATA_DIR / "mock_labels.csv", index=False)

# nu maken we nep visuele embeddings
# voor elke patiënt maken we 5 vectoren van lengte 768
# dit bootst bijvoorbeeld meerdere beeldpatches per patiënt na
vis_data = {p: [np.random.randn(768) for _ in range(5)] for p in pids}

# daarnaast maken we ook tekst-embeddings
# hier krijgt elke patiënt precies één vector van lengte 768
txt_data = {p: np.random.randn(768) for p in pids}

# sla de visuele embeddings op als pickle-bestand
# pickle gebruiken we omdat numpy-arrays zo makkelijk opgeslagen kunnen worden
pickle.dump(vis_data, open(DATA_DIR / "mock_visual.pkl", "wb"))

# sla de tekst-embeddings ook op als pickle-bestand
pickle.dump(txt_data, open(DATA_DIR / "mock_text.pkl", "wb"))

# simpele bevestiging dat alles gelukt is
print("Mock data generated in data/")
