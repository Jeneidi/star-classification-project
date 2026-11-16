from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
manifest = pd.read_csv(root / "data" / "manifest.csv")
v = pd.read_csv(root / "data" / "vband_catalog.csv").set_index("asassn_name")
g = pd.read_csv(root / "data" / "gband_catalog.csv").set_index("ID")

v_names = manifest.vband_files.str.rsplit("/", n=1).str[-1].str.removesuffix(".dat").str.replace("ASASSN-VJ", "ASASSN-V J", regex=False)
g_names = manifest.gband_files.str.rsplit("/", n=1).str[-1].str.removesuffix(".dat").str.replace("ASASSN-V_", "ASASSN-V ", regex=False)
v = v.loc[v_names].reset_index(drop=True)
g = g.loc[g_names].reset_index(drop=True)

features = pd.DataFrame({
    "v_mag": v.mean_vmag,
    "g_mag": g.Mean_gmag,
    "v_amplitude": v.amplitude,
    "g_amplitude": g.Amplitude,
    "v_log_period": np.log10(v.period.where(v.period > 0)),
    "g_log_period": np.log10(g.Period.where(g.Period > 0)),
    "v_lksl": v.lksl_statistic,
    "g_lksl": g.LKSL_statistic,
    "bp_rp": v.bp_rp,
    "j_k": v.j_k,
    "w1_w2": v.w1_w2,
})

X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(features))
correlations = pd.DataFrame(X, columns=features.columns).corr()
print(correlations.round(2))
pca = PCA(n_components=8)
coords = pca.fit_transform(X)

result = manifest[["star", "v_class", "g_class"]].copy()
#PCA 1-8,  this basically decides our inputs for MLP, we could do 11, or mayby this is more efficient (could try)
#          CNN probably just give it the light curve
#why 8? vibes
for i in range(8):
    result[f"PC{i + 1}"] = coords[:, i]

result.to_csv(root / "pca_scores.csv", index=False)

classes = manifest.v_class.where(
    manifest.v_class.isin(manifest.v_class.value_counts().head(10).index),
    "OTHER"
)
#plot pca
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, i in zip(axes.flat, range(0, 8, 2)):
    for name in classes.unique():
        mask = classes == name
        ax.scatter(coords[mask, i], coords[mask, i + 1],
                   s=8, alpha=0.7, label=name)

    ax.set_xlabel(f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.1%})")
    ax.set_ylabel(f"PC{i + 2} ({pca.explained_variance_ratio_[i + 1]:.1%})")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)

plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig(root / "pca.png", dpi=400)
#plt.show()

print("Variance explained:", pca.explained_variance_ratio_)
print(pd.DataFrame(
    pca.components_.T,
    index=features.columns,
    columns=[f"PC{i + 1}" for i in range(8)]
).round(3))

#plot correlation
plt.figure(figsize=(10, 8))
plt.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(11), features.columns, rotation=90)
plt.yticks(range(11), features.columns)
plt.colorbar()
plt.tight_layout()
plt.savefig(root / "correlations.png", dpi=400)