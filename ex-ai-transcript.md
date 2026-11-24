# AI transcript: the CNN part of the star-classification repo

---

## Stage 1: Set up the task

**PROMPT 1**
> Group project, classifying variable stars from ASAS-SN light curves. The data is already sampled and split by teammates: `data/train.csv`, `val.csv`, `test.csv`, each row has the star, its class, and paths to a V-band and a g-band `.dat` file. There is also an MLP script. I own the CNN. Aiden designed it: Conv1d 32 filters kernel 9, max pool, Conv1d 64 filters kernel 9, global average pool, dense 64, softmax. I need to compare MLP vs this CNN vs the CNN with FGSM adversarial training, and test whether a model trained on one band still works on the other. Where do I start?

**REPLY 1**
> Plan for one script, `cnn.py`:
> 1. Read and clean the `.dat` light curves.
> 2. Turn each curve into a fixed-length input the CNN can read.
> 3. Define MLP, CNN and CNN+FGSM so all three get the same input.
> 4. Train each on one band, test on both bands.
> 5. Save tables, figures and the best model per setting.
>
> Decide the comparison rules before writing any model code, because they decide whether the result means anything. I would fix: same stars for every model and band pair, same input for all three, early stopping that never looks at the other band, several seeds, and a class-balanced loss. Want me to write it with those?

**PROMPT 2**
> Yes. The older MLP just stretches the raw curve to 128 points. Should the CNN do that too?

**REPLY 2**
> No. Stretching a curve to 128 points in time order mixes years of observations into a noisy cloud. A periodic star's shape only shows when you **fold** the curve on its period: put every observation at its phase (time divided by period, keep the remainder) and take the median in 128 phase bins. A CNN then sees one clean cycle.
>
> Inputs per star and band: 128 folded bins (magnitude minus the star's median, so brightness offsets do not matter), `log10(period)`, and a has-period flag.

---

## Stage 2: Cleaning and folding

**PROMPT 3**
> Some `.dat` files have no light curves at all and the MLP script just skips them. What should the CNN script do?

**REPLY 3**
> Be explicit and log it. In `read_curve`, drop what is not a real measurement:
> - rows with no magnitude,
> - upper limits such as `>17.4` (they become NaN),
> - points with `mag_err = 99.99` (bad measurements).
>
> ```python
> good = np.isfinite(t) & np.isfinite(m) & (m > 0) & (m < 30) & np.isfinite(e) & (e < 1)
> ```
> Then require at least 20 good points in **both** bands (`MIN_POINTS = 20`). Keeping a star only when both bands work means every model and every train-to-test band pair is scored on the same stars. Write every dropped star and the reason to `results/cnn_dropped_stars.csv` so nobody can say you quietly removed hard examples. Header-only V files (about 90) end up there.

**PROMPT 4**
> Irregular `L` stars have no period. What do I fold them on?

**REPLY 4**
> Fold them over their own time span, so they are binned over time instead of by phase:
> ```python
> has_period = bool(np.isfinite(period) and period > 0)
> p = period if has_period else (t.max() - t.min()) * 1.000001 + 1e-6
> phase = ((t - t.min()) / p) % 1.0
> ```
> The has-period flag tells the model which case it is looking at. Periodic curves are also rotated so the brightest bin (smallest magnitude) comes first, and empty bins are filled by circular interpolation, since phase 1 wraps to phase 0.

---

## Stage 3: Models and fairness rules

**PROMPT 5**
> Write the three models. Aiden's CNN should use circular padding if that makes sense.

**REPLY 5**
> It does: a folded curve is a loop, so the left edge should see the right edge.
> ```python
> self.features = nn.Sequential(
>     nn.Conv1d(1, 32, kernel_size=9, padding=4, padding_mode="circular"), nn.ReLU(),
>     nn.MaxPool1d(2),
>     nn.Conv1d(32, 64, kernel_size=9, padding=4, padding_mode="circular"), nn.ReLU(),
>     nn.AdaptiveAvgPool1d(1), nn.Flatten())
> self.head = nn.Sequential(
>     nn.Linear(64 + 2, 64), nn.ReLU(), nn.Dropout(0.2),
>     nn.Linear(64, n_classes))
> ```
> The convolutions slide filters along the curve to find local patterns (dips, rises, flat parts); max pooling halves the length; the second convolution finds patterns of patterns; global average pooling summarizes them; the dense layer adds the two period inputs and combines everything. The MLP baseline gets the same 130 inputs (128 bins plus 2 period values): `130 -> 128 -> 64 -> 32 -> n_classes`, ReLU, dropout 0.2.

**PROMPT 6**
> Explain FGSM in plain words and add it. Use 0.05 mag as the training step.

**REPLY 6**
> FGSM (fast gradient sign method) asks: which tiny nudge to every input value would hurt the model most? Take the gradient of the loss with respect to the **input**, then move each bin by `eps` in the direction of that gradient's sign:
> ```python
> def fgsm(model, curve, extra, y, eps, loss_fn):
>     curve = curve.clone().requires_grad_(True)
>     grad, = torch.autograd.grad(loss_fn(model(curve, extra), y), curve)
>     return (curve + eps * grad.sign()).detach()
> ```
> Adversarial training mixes attacked curves into every batch:
> ```python
> x_adv = fgsm(model, x[b], z[b], y[b], EPS_TRAIN, loss_fn)
> loss = 0.5 * loss + 0.5 * loss_fn(model(x_adv, z[b]), y[b])
> ```
> Half clean loss, half adversarial loss. The units are magnitudes, so 0.05 means every bin moves by 0.05 mag.

**PROMPT 7**
> How do I keep the cross-band test honest?

**REPLY 7**
> Four rules in `train()`:
> 1. **Early stopping on validation macro-F1 in the training band only.** The other band is never looked at during training or model selection.
> 2. **Standardize the period inputs with training statistics only**, in the training band.
> 3. **Class-balanced loss** (`compute_class_weight("balanced")`) so EW and SR do not dominate.
> 4. **5 seeds** (0 to 4). Report mean and standard deviation, because gaps of 0.01 to 0.02 sit inside the noise.
>
> Also score only classes that appear in both val and test. Types with fewer than 7 stars live in train only, so they cannot be scored and are left out; say so in the report.

**PROMPT 8**
> It was still improving at epoch 200, so I want more epochs. How long will 30 runs take?

**REPLY 8**
> Set `MAX_EPOCHS = 500` with `PATIENCE = 40` (stop when validation macro-F1 has not improved for 40 epochs). Total is 3 models x 2 bands x 5 seeds = 30 training runs. Print progress and an ETA after every run. On one GPU expect about 6 minutes, on a laptop CPU about 20.

---

## Stage 4: Robustness tests

**PROMPT 9**
> The proposal says we test noise robustness too. What levels, and how is that different from FGSM?

**REPLY 9**
> Two different questions:
> - **Gaussian noise:** random jitter on each bin, std 0, 0.01, 0.03, 0.10, 0.20 mag. Answers "does it survive messy data?"
> - **FGSM attack at test time:** the worst-case nudge for that specific model, size 0.01, 0.03, 0.05, 0.10 mag. Answers "can a deliberate perturbation fool it?"
>
> Run both for every trained model on both bands and save everything to `results/cnn_all_runs.csv`. Use a fixed random generator per seed (`1000 + seed`) so the noise is reproducible.

---

## Stage 5: Run it and read the numbers

**PROMPT 10**
> Ran it on the FSU GPU, 5 min 52 s on an A4500. Here is `cnn_summary.csv`. What can I say?

**REPLY 10**
> Test set, 595 stars, mean over 5 seeds, macro-F1 / accuracy:
>
> | model | V to V | V to G | G to G | G to V |
> |---|---|---|---|---|
> | MLP | 0.908 / 0.929 | 0.834 / 0.914 | 0.860 / 0.916 | 0.863 / 0.906 |
> | CNN | 0.906 / 0.941 | 0.824 / 0.928 | 0.848 / 0.910 | 0.809 / 0.877 |
> | CNN+FGSM | 0.893 / 0.906 | 0.853 / 0.901 | 0.857 / 0.885 | 0.814 / 0.871 |
>
> What you can claim:
> - On clean curves all three are within about one standard deviation. Do **not** say the CNN wins. It has the best V accuracy (0.941), and CNN+FGSM has the best V to G macro-F1 (0.853), but the gaps are small.
> - Going to the other band costs roughly 0.05 to 0.08 macro-F1 for most models.
> - The real separation is robustness. Under an FGSM attack of 0.05 mag (V to V macro-F1): CNN+FGSM 0.70, CNN 0.48, MLP 0.32.
> - Random noise hurts the plain CNN most. At 0.2 mag, V to V macro-F1: MLP 0.68, CNN+FGSM 0.60, CNN 0.49.

**PROMPT 11**
> The confusion matrix shows `L` stars trained on V get called `SR` in g. Is that a bug?

**REPLY 11**
> Not a code bug, a catalog difference. In the V catalog 0 of 31 training `L` stars have a period, so the model learns "no period means L". In the g catalog 24 of 31 `L` stars do have a period, so they get folded like semi-regular stars and look like `SR`. The model is behaving correctly on the input it was given.
>
> Put this in the discussion as the main cross-band error, with `pictures/cnn_confusion_cross_band.png` as evidence. It is a more interesting finding than a clean table: cross-band transfer fails where the two catalogs treat the same star differently.

---

## Stage 6: Outputs, README, limitations

**PROMPT 12**
> Make the script save everything the report needs and write the README section so teammates can rerun it.

**REPLY 12**
> Outputs:
> - `results/cnn_summary.csv` (accuracy and macro-F1, mean and std), `cnn_per_class.csv`, `cnn_all_runs.csv`, `cnn_training.csv`, `cnn_dropped_stars.csv`
> - `pictures/cnn_lightcurve_examples.png` and `cnn_class_median_curves.png` (what the inputs look like), `cnn_cross_band_f1.png`, `cnn_noise_robustness.png`, `cnn_fgsm_robustness.png`, `cnn_confusion_cross_band.png`
> - `models/cnn_*.pt` for seed 0 of each model and band
>
> README section: the one command (`python cnn.py`), runtime estimate, libraries (`torch numpy pandas scikit-learn matplotlib`), input description, cleaning rules, class list, model table, training settings, output files, results table.

**PROMPT 13**
> What limitations should the report admit for my part?

**REPLY 13**
> - Only 12 classes are scored; types with fewer than 7 stars are train-only.
> - One split, 5 seeds: no confidence interval over different splits.
> - Adversarial results use FGSM only, the weakest standard attack. Do not claim general adversarial robustness.
> - Perturbations are in magnitudes on a folded curve, so a real adversary changing raw observations is a different threat model.
> - CNN+FGSM trades a little clean performance for robustness: its V to V macro-F1 (0.893) is below the CNN's (0.906), though within the spread.

---

## Order of work for this part

1. Settle comparison rules (same stars, same input, honest cross-band test)
2. Clean and fold the light curves (`read_curve`, `fold`)
3. Define MLP, CNN and CNN+FGSM
4. Train 30 runs with early stopping, then noise and FGSM tests
5. Read the numbers conservatively, explain the L to SR error
6. Save outputs, write the README section, list limitations
