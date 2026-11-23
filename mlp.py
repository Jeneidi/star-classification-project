from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

root = Path(__file__).resolve().parent
data = root / "data"
(root / "models").mkdir(exist_ok=True)

torch.manual_seed(42)
train = pd.read_csv(data / "train.csv")
test = pd.read_csv(data / "test.csv")
classes = sorted(train.v_class.unique())
class_ids = {label: i for i, label in enumerate(classes)}

#some .dat files have no curves woops.. just exclude them 
def curve(filename, band):
    df = pd.read_csv(data / filename, sep=r"\s+", skiprows=1 if band == "v" else 0)
    col = "MAG" if band == "v" else "mag"
    mag = pd.to_numeric(df.sort_values("HJD")[col], errors="coerce").to_numpy()
    mag = mag[np.isfinite(mag) & (mag > 0) & (mag < 30)]
    if not len(mag):
        print("Empty light curve:", filename)
        return None
    mag = np.interp(np.linspace(0, len(mag) - 1, 128), np.arange(len(mag)), mag)
    return (mag - np.median(mag)).astype("float32")


def load_curves(rows):
    g, v, labels, keep = [], [], [], []
    for i, star in enumerate(rows.itertuples()):
        gc = curve(star.gband_files, "g")
        vc = curve(star.vband_files, "v")
        if gc is None or vc is None:
            continue
        g.append(gc)
        v.append(vc)
        labels.append(class_ids[star.v_class])
        keep.append(i)
    return {"g": np.stack(g), "v": np.stack(v)}, torch.tensor(labels), rows.iloc[keep]


# prepare experiment, catalog provided data vs light curves
light_train, y_train, train_rows = load_curves(train)
light_test, y_test, test_rows = load_curves(test)

v = pd.read_csv(data / "vband_catalog.csv").set_index("asassn_name")
g = pd.read_csv(data / "gband_catalog.csv").set_index("ID")


def catalog(rows, band):
    names = rows[f"{band}band_files"].str.rsplit("/", n=1).str[-1].str.removesuffix(".dat")
    if band == "v":
        names = names.str.replace("ASASSN-VJ", "ASASSN-V J", regex=False)
        x = v.loc[names, ["amplitude", "mean_vmag", "period"]].to_numpy(dtype="float32")
    else:
        names = names.str.replace("ASASSN-V_", "ASASSN-V ", regex=False)
        x = g.loc[names, ["Amplitude", "Mean_gmag", "Period"]].to_numpy(dtype="float32")
    x[:, 2] = np.log10(np.where(x[:, 2] > 0, x[:, 2], np.nan))
    x[~np.isfinite(x)] = np.nan
    return x


catalog_train = {band: catalog(train_rows, band) for band in ("g", "v")}
catalog_test = {band: catalog(test_rows, band) for band in ("g", "v")}

#kinda obfuscated,

# takes the time series in the .dat as input 
for band in ("g", "v"):
    x = torch.tensor(light_train[band], dtype=torch.float32)

    lightcurve_model = nn.Sequential(
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, len(classes)),
    )

    optimizer = torch.optim.Adam(lightcurve_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        optimizer.zero_grad()
        loss = criterion(lightcurve_model(x), y_train)
        loss.backward()
        optimizer.step()
        print(f"light {band} | epoch {epoch} | loss {loss.item():.4f}")

    torch.save(
        {"weights": lightcurve_model.state_dict(), "classes": classes},
        root / "models" / f"mlp_{band}.pt"
    )

    lightcurve_model.eval()
    with torch.no_grad():
        for test_band in ("g", "v"):
            x_test = torch.tensor(light_test[test_band], dtype=torch.float32)
            predictions = lightcurve_model(x_test).argmax(dim=1)
            accuracy = (predictions == y_test).float().mean().item()
            print(f"light {band} -> {test_band}: {accuracy:.3f}")


# catalog mlp (takes provided amp,mag,period as inputs
for band in ("g", "v"):
    x = catalog_train[band].copy()

    median = np.nan_to_num(np.nanmedian(x, axis=0))
    x = np.where(np.isnan(x), median, x)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1
    x = (x - mean) / std

    x = torch.tensor(x, dtype=torch.float32)

    catalog_model = nn.Sequential(
        nn.Linear(3, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, len(classes)),
    )

    optimizer = torch.optim.Adam(catalog_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        optimizer.zero_grad()
        loss = criterion(catalog_model(x), y_train)
        loss.backward()
        optimizer.step()
        print(f"catalog {band} | epoch {epoch} | loss {loss.item():.4f}")

    torch.save(
        {
            "weights": catalog_model.state_dict(),
            "classes": classes,
            "median": median,
            "mean": mean,
            "std": std
        },
        root / "models" / f"catalog_mlp_{band}.pt"
    )

    catalog_model.eval()
    with torch.no_grad():
        for test_band in ("g", "v"):
            x_test = catalog_test[test_band]
            x_test = np.where(np.isnan(x_test), median, x_test)
            x_test = (x_test - mean) / std

            x_test = torch.tensor(x_test, dtype=torch.float32)
            predictions = catalog_model(x_test).argmax(dim=1)
            accuracy = (predictions == y_test).float().mean().item()
            print(f"catalog {band} -> {test_band}: {accuracy:.3f}")


