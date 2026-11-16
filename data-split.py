from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
manifest = pd.read_csv(root / "data" / "manifest.csv")

competing = manifest[manifest.v_class != manifest.g_class].copy()
competing["classification_status"] = "competing classification by ASAS-SN"
competing.to_csv(root / "competing_classifications.csv", index=False)

agreed = manifest[manifest.v_class == manifest.g_class].copy()
agreed["split"] = "train"
rng = np.random.default_rng(42)

for _, group in agreed.groupby("v_class"):
    indices = rng.permutation(group.index.to_numpy())
    if len(indices) < 7:
        continue
    n = round(len(indices) * 0.15)
    agreed.loc[indices[:n], "split"] = "val"
    agreed.loc[indices[n:2 * n], "split"] = "test"

for name in ("train", "val", "test"):
    agreed[agreed.split == name].to_csv(root / f"{name}.csv", index=False)

#g and v band disagree sometimes, could be interesting
print("Competing classifications:", len(competing))
print(agreed.split.value_counts())
print(pd.crosstab(agreed.v_class, agreed.split))
