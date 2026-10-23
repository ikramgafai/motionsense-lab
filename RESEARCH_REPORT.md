# MotionSense Lab: Research Report

**Lightweight human activity recognition with accelerometer/gyroscope fusion on UCI HAR**

All numbers below were measured on the study machine and are read from
`results/` (main run) and `results/supplementary/` (post-hoc sensitivity runs).
`scripts/audit_results.py` re-derives them from saved artifacts (352 checks on
the main run, 0 failures; see §8).

---

## 0. Answer in one paragraph

On 9 unseen test subjects, a ~25k-parameter 1D CNN reached **0.912 ± 0.006 test
macro-F1 with accelerometer + gyroscope** (9 channels; mean ± SD over 5 seeds),
compared with **0.902 ± 0.008 with accelerometer channels only** and **0.788 ±
0.017 with gyroscope only**. Fusion vs gyroscope-only: a large gain (+0.124),
consistent on 5/5 seeds and 8/9 subjects. The subject-level Wilcoxon test
narrowly misses significance after multiple-comparison correction (Holm p =
0.059) because 9 subjects give the test little power. Fusion vs
accelerometer-only: a small gain (+0.010 macro-F1 in the main run, +0.013 on
a second validation split), positive on most seeds. With 9 test subjects it
is not distinguishable from subject-to-subject variation: the Wilcoxon p-value is 0.20 (Holm-adjusted
0.26), and one subject accounts for most of the per-subject gain. Fusion costs
+672 parameters (+2.7 %) and adds no latency difference measurable with this
setup. **Model-only CPU inference** took about 0.5 ms median (0.77–0.88 ms
p95) per already-recorded window on one thread for every CNN variant. This is
not end-to-end recognition latency. A window must first be filled with 2.56 s
of signal, so recognition delay is dominated by the observation window, not
the model (§5.3). Conclusion: fusion is cheap and does not hurt on clean data. A
meaningful accuracy gain over a good accelerometer model is **not
established** by this study. One new risk was measured: the fused model
degrades *below* the accelerometer-only model when the gyroscope becomes very
noisy.

**Scope.** This is inertial-sensor research: a waist-mounted smartphone's
accelerometer and gyroscope. It involves no cameras, video or pose
estimation, and its results should not be read as evidence about
camera-based motion tracking.

---

## 1. Hypotheses

The research question was fixed before any experiment was run: *does combining
accelerometer and gyroscope signals improve activity recognition for unseen
subjects while maintaining efficient inference?* It was split into two
testable hypotheses:

* **H1 (accuracy).** For a fixed model family and training budget, the fused
  input (acc+gyro) gives higher test macro-F1 on unseen subjects than either
  sensor alone.
* **H2 (efficiency).** The fused model's CPU inference cost (parameters,
  latency at batch 1) is not materially higher than the single-sensor models,
  and is small relative to the 2.56 s observation window.

Primary metric: **macro-F1** on the official test subjects. The classes are
nearly balanced, but macro-F1 weights the hard static postures equally with the
easy ones. Accuracy is also reported.

## 2. Dataset

* **Source:** official UCI archive
  `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`,
  downloaded by `python -m motionsense download`. The download is checked
  against SHA-256
  `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
* **Citation and terms:** the dataset authors require citing D. Anguita,
  A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz, *A Public Domain Dataset
  for Human Activity Recognition Using Smartphones*, 21st European Symposium
  on Artificial Neural Networks, Computational Intelligence and Machine
  Learning (ESANN 2013), Bruges, Belgium. Their license text states that the
  data is distributed as-is and that commercial use is prohibited. This
  project downloads the data from UCI at run time and does not redistribute
  it.
* **Nested ZIP:** the outer archive contains `UCI HAR Dataset.zip`. It is read
  in memory, and only a whitelist of 23 files is extracted (raw inertial
  signals, labels, subject ids). This also rules out path traversal from
  archive members.
* **Input:** raw inertial windows, **128 samples × 9 channels** at 50 Hz (2.56
  s, 50 % overlap between consecutive windows). The precomputed 561-feature
  vectors are **not used**.
* **Size:** 7,352 training windows from 21 subjects and 2,947 test windows from
  9 subjects. Six activities: walking, upstairs, downstairs, sitting, standing,
  laying. All 9 test subjects have all 6 activities (checked by the audit), so
  per-subject macro-F1 is well defined.

**Channel semantics matter for the ablation.** `total_acc_{x,y,z}` is the
accelerometer in g. `body_acc_{x,y,z}` is *derived from `total_acc`* by the
dataset authors: per the dataset description, gravity is estimated with a 0.3
Hz Butterworth low-pass filter and `body_acc` = `total_acc` − gravity (in
effect a high-pass). It is not a second sensor. Both sensors were also
"pre-processed by applying noise filters" before windowing; the `.names` file
does not say which. `body_gyro_{x,y,z}` is the gyroscope in rad/s. So
"accelerometer only" has 6 channels from one physical sensor, while "gyroscope
only" has 3. To separate *sensor* from *channel count*, two control conditions
were added:

| condition | channels | n |
|---|---|---|
| `acc` | total_acc + body_acc | 6 |
| `gyro` | body_gyro | 3 |
| `acc_gyro` | all | 9 |
| `acc_raw3` (control) | total_acc only | 3 |
| `acc_raw3_gyro` (control) | total_acc + body_gyro | 6 |

## 3. Leakage prevention

| risk | mitigation | verified by |
|---|---|---|
| Same subject in train and test | official subject-disjoint split kept as is | `data.validate`, audit §1 |
| Same subject in train and validation (windows overlap 50 % within a recording) | validation = 5 whole subjects drawn (seed 0) from the 21 training subjects: {8, 11, 16, 19, 23}; training = the other 16 | unit test `test_subject_split_is_disjoint_and_complete`; audit checks every run |
| Normalisation statistics leaking val/test information | per-channel mean/SD fitted on the 16-subject training partition only; val and test are only transformed | unit test with shifted val/test data; audit recomputes every scaler and confirms it differs from train+val statistics |
| Test-set hyperparameter selection | LogReg `C` and CNN early-stopping epoch selected on validation macro-F1; `train_cnn` never receives test data; test is evaluated once per final model | code structure; audit checks best epoch = argmax of the validation curve |
| BatchNorm statistics | running stats from training batches only; eval mode for val/test | `model.eval()` in `predict` |
| Robustness experiment re-tuning | noise applied only at test time to already-final models; nothing re-selected | `robustness.py` loads saved weights |

One disclosure: I looked at the main-run test results before designing the two
supplementary analyses (§6) and the Holm correction (§4.4). Those additions
cannot leak test information into the main models, which were already final.
But they are post-hoc and are labelled as such. The supplementary analyses
only *weaken* or *confirm* claims; neither one was used to pick a "better"
headline number.

## 4. Methodology

### 4.1 Models

**Logistic regression baseline** (`sklearn.linear_model.LogisticRegression`,
lbfgs, multinomial, L2). Input is the standardised window flattened to
128·C features. `C` is chosen from {0.001, 0.01, 0.1, 1} by validation
macro-F1. This is deliberately the weakest reasonable learned baseline. It
gives each (channel, timestep) its own fixed weight. It can therefore exploit
class differences in mean signal level (notably gravity orientation) and
waveform patterns that recur at a consistent position in the window. It has
no mechanism to pool over the phase at which a gait cycle starts, and as a
linear score it cannot compute nonlinear summaries such as signal energy or
dominant frequency. It is deterministic, so it was run once per condition.

**1D CNN** (`models.SmallCNN`): three blocks of Conv1d → BatchNorm → ReLU
(widths 32/64/64, kernels 7/5/3, max-pool ×2 after the first two), global
average pooling, dropout 0.3, then a linear layer to 6 classes. It has
23,910–25,254 parameters; only the first conv layer depends on the channel
count. That is about 1.13M / 1.22M / 1.31M multiply-accumulates for 3 / 6 / 9
input channels (analytic count of conv and linear layers). Why this
architecture: convolution plus global pooling gives the time-shift tolerance
that the linear model lacks. It is small enough for CPU or edge inference. And
the identical body across conditions makes the sensor ablation a controlled
comparison: only the input differs.

### 4.2 Training

All hyperparameters were fixed in `configs/default.yaml` before the main run
and are **the same for every sensor condition** (equal tuning budget):
AdamW (lr 1e-3, weight decay 1e-4), batch 64, cross-entropy, at most 60
epochs, early stopping with patience 10 on validation macro-F1, and the best
validation checkpoint restored. 5 seeds (0–4) per condition. Seeding covers
Python, NumPy and torch, with `torch.use_deterministic_algorithms(True)` and a
seeded shuffling generator. A retrain of `cnn_acc_gyro_s1` reproduced the best
epoch and test macro-F1 (0.915665) bit-for-bit. Each run stores its resolved
config, split subjects, scaler statistics, learning curve, confusion matrix and
per-subject metrics.

### 4.3 Latency protocol (`benchmark.py`)

* A separate stage, run after all training finished, so nothing else was
  loading the CPU.
* Batch size 1, a single real test window, `torch.inference_mode()`, eval mode.
* 100 warm-up calls, then 500 timed calls (`perf_counter_ns`) per round, 3
  rounds with the order of the 90 (run, measurement) jobs shuffled each round (the 90 jobs are 25 CNN runs × 3 measurements plus 5 LogReg runs × 3).
  Samples are pooled, then median and p95 taken. Tables report the mean over
  the 5 seeds of each run's median and p95.
* Threads: torch intra-op threads and BLAS/OpenMP (`threadpoolctl`) limited to
  1 for the primary number; 4 threads also measured for the CNN.
* What is timed:
  * `model`: forward pass on an already-standardised window.
  * `e2e`: raw window → standardise → model → argmax.
  * For LogReg, also `numpy`: bare matmul + argmax, which separates the
    arithmetic from scikit-learn's per-call validation overhead.
* What is **not** timed: the 2.56 s needed to *observe* a window, sensor I/O,
  and app scheduling. In a streaming deployment, decision delay is the
  observation window (or the 1.28 s hop at 50 % overlap) plus the inference
  time below.
* Machine: Windows 11, AMD Family 25 Model 117 (Zen 4 laptop CPU), 16 logical
  cores, Python 3.13.7, torch 2.12.0+cpu, scikit-learn 1.7.2.

### 4.4 Statistics

The unit of generalisation is the **test subject**, not the window. Windows
from one person are highly correlated and overlap by 50 %, so window-level
significance tests would badly overstate certainty. For each pair of
conditions, `analysis.py` reports:

* seed-level differences in overall test macro-F1, using the same seeds in
  both arms;
* per-subject differences in macro-F1 (averaged over seeds), with a
  two-sided Wilcoxon signed-rank test, a percentile-bootstrap 95 % CI over the
  9 subjects (10k resamples), and a count of subjects that improved;
* Holm-adjusted p-values within each model family (5 comparisons each).

With n = 9 subjects, the smallest possible two-sided exact Wilcoxon p is
0.0039, reached only when all 9 differences point the same way. The p-value
depends on the ranks of the differences, not just how many are positive.
Fusion vs gyro (8/9 subjects better, the one negative difference being S10)
gives p = 0.0117, which does not survive Holm correction across 5 tests.
Zero differences are dropped (scipy's default `zero_method="wilcox"`). In the
fusion-vs-acc comparison, S24 scores 1.000 macro-F1 in both conditions, so
that test uses n = 8. All stored p-values were re-checked by exact
enumeration of the sign-flip distribution (`scripts/verify_report_claims.py`).
Percentile-bootstrap intervals over 9 points tend to be too narrow (they
under-cover at small n). Subject-level claims here are therefore inherently
low-powered. A non-significant result means "not shown", not "no effect".

## 5. Results (main run, `configs/default.yaml`)

### 5.1 Accuracy, macro-F1, size and latency

**1D CNN**: mean ± SD over 5 seeds. Latency in ms, 1 thread unless stated.

| condition | ch | params | val macro-F1 | test accuracy | **test macro-F1** | model median / p95 | 4-thread median / p95 | e2e median / p95 |
|---|---|---|---|---|---|---|---|---|
| acc | 6 | 24,582 | 0.910 | 0.902 ± 0.007 | **0.902 ± 0.008** | 0.454 / 0.768 | 0.516 / 0.881 | 0.544 / 0.917 |
| gyro | 3 | 23,910 | 0.814 | 0.784 ± 0.017 | **0.788 ± 0.017** | 0.514 / 0.849 | 0.513 / 0.859 | 0.553 / 0.927 |
| acc+gyro | 9 | 25,254 | 0.913 | 0.911 ± 0.006 | **0.912 ± 0.006** | 0.504 / 0.832 | 0.529 / 0.896 | 0.561 / 0.924 |
| total_acc (ctrl) | 3 | 23,910 | 0.912 | 0.891 ± 0.014 | **0.891 ± 0.014** | 0.506 / 0.805 | 0.511 / 0.842 | 0.522 / 0.876 |
| total_acc+gyro (ctrl) | 6 | 24,582 | 0.912 | 0.913 ± 0.009 | **0.912 ± 0.010** | 0.512 / 0.882 | 0.493 / 0.918 | 0.551 / 0.901 |

**Logistic regression**: one deterministic run per condition. Latency in ms,
1 thread.

| condition | ch | params | selected C | val macro-F1 | test accuracy | **test macro-F1** | sklearn predict median / p95 | NumPy matmul median / p95 |
|---|---|---|---|---|---|---|---|---|
| acc | 6 | 4,614 | 0.001 | 0.565 | 0.544 | **0.504** | 0.069 / 0.134 | 0.0075 / 0.0130 |
| gyro | 3 | 2,310 | 0.01 | 0.290 | 0.305 | **0.292** | 0.051 / 0.099 | 0.0074 / 0.0109 |
| acc+gyro | 9 | 6,918 | 0.001 | 0.585 | 0.561 | **0.529** | 0.064 / 0.146 | 0.0081 / 0.0089 |
| total_acc (ctrl) | 3 | 2,310 | 0.001 | 0.579 | 0.507 | **0.450** | 0.072 / 0.133 | 0.0074 / 0.0121 |
| total_acc+gyro (ctrl) | 6 | 4,614 | 0.001 | 0.582 | 0.547 | **0.515** | 0.063 / 0.133 | 0.0082 / 0.0130 |

Figures: [comparison_macro_f1.png](results/figures/comparison_macro_f1.png),
[efficiency.png](results/figures/efficiency.png). Tables:
[summary.csv](results/summary.csv), [runs.csv](results/runs.csv).

**Baseline comparison.** The CNN beats flattened-window LogReg by 0.38–0.50
macro-F1 in every condition. The LogReg per-class F1 shows why: it does well
on laying (0.96 with accelerometer input), where static gravity direction is
the signal, but it is poor on the three walking classes (0.07–0.47). Their
signature is periodic, and the gait phase at the window start varies from
window to window. A fixed per-timestep weight vector can match only
phase-locked templates, and a linear score cannot compute phase-independent
summaries such as energy or dominant frequency. We attribute most of the
baseline gap to this missing shift tolerance and nonlinearity rather than to
model size. That attribution was not isolated experimentally: a feature-based
or shift-augmented linear baseline would be needed to test it (§10). LogReg has fewer parameters than the
CNN (2.3k–6.9k vs 24k–25k), but it is not "competitive at a smaller size". It
answers a different, much easier question.

### 5.2 Sensor ablation: paired comparisons (CNN)

Positive Δ = first condition better. Seed Δ = difference in overall test
macro-F1 (mean ± SD over 5 paired seeds). Subject Δ = mean over 9 test
subjects of the per-subject macro-F1 difference. Source:
[comparisons.json](results/comparisons.json).

| comparison | seed Δ (seeds better) | subject Δ [95 % bootstrap CI] | subjects better | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| acc+gyro vs acc (9 vs 6 ch) | +0.0095 ± 0.0127 (4/5) | +0.0145 [−0.0058, +0.0407] | 6/9 | 0.195 | 0.258 |
| acc+gyro vs gyro (9 vs 3 ch) | +0.1241 ± 0.0212 (5/5) | +0.1198 [+0.0624, +0.1644] | 8/9 | 0.012 | 0.059 |
| total_acc+gyro vs total_acc (6 vs 3 ch) | +0.0214 ± 0.0182 (4/5) | +0.0296 [+0.0038, +0.0594] | 7/9 | 0.074 | 0.223 |
| total_acc vs gyro (3 vs 3 ch) | +0.1031 ± 0.0232 (5/5) | +0.0922 [+0.0378, +0.1403] | 7/9 | 0.020 | 0.078 |
| acc vs total_acc (adds derived body_acc) | +0.0115 ± 0.0174 (3/5) | +0.0131 [−0.0007, +0.0320] | 6/9 | 0.129 | 0.258 |

"Subjects better" counts strict improvements. For acc+gyro vs acc, the 9
subjects split into 6 better, 2 worse, and 1 tie: S24 scores 1.000 in both
conditions.

LogReg, for reference (single run, so no seed variance):

* acc+gyro vs acc +0.025 (subject CI [+0.011, +0.045], 8/9 subjects, Holm p 0.039)
* acc+gyro vs gyro +0.237 (9/9, Holm p 0.020)
* total_acc+gyro vs total_acc +0.065 (9/9, Holm p 0.020)

These LogReg gaps are sensitive to the choice of C (§6A).

**Reading.**

1. *Accelerometer > gyroscope* at equal channel count (3 vs 3: +0.10 macro-F1,
   5/5 seeds, 7/9 subjects; Holm p = 0.078, so not significant at the subject
   level after correction). The confusion matrices are consistent with the physics: a
   gyroscope measures rotation rate and gives no direct gravity cue in a
   stationary posture, and the gyro-only model largely fails to separate
   **laying** from sitting/standing (laying recall 0.59; 25 % of laying
   windows predicted as sitting), and its sit/stand
   separation is worse too.
2. *Adding the gyroscope to the accelerometer* helps most when the
   accelerometer input is minimal: +0.021 seed Δ for raw 3-channel total_acc,
   and +0.010 once the derived body_acc channels are present. The per-class
   gain sits in the **dynamic classes** (raw3 comparison: walking +0.055 F1,
   upstairs +0.037) and in sitting (+0.041). Downstairs (−0.020) and laying
   (−0.014) drop slightly.
3. Adding the *derived* body_acc channels to total_acc gives a similar small
   gain (+0.0115). That is plausible: a fixed linear filter (total minus
   low-passed gravity) is something a CNN could in principle learn. Its benefit is partly an easier input
   representation, not new information.

### 5.3 Efficiency (H2)

* Fusion adds only first-layer weights: +672 parameters vs acc (+2.7 %) and
  +1,344 vs gyro. That is about 7 % / 15 % more MACs.
* **Latency differences between CNN conditions are not measurable with this
  setup.** The spread of median latency *across seeds of the same
  architecture* (0.40–0.62 ms) is larger than the differences between
  conditions (0.45–0.51 ms means). Identical architectures with different
  weights should cost the same, so that spread is measurement noise (CPU
  frequency, cache, OS scheduling). It sets the resolution of this
  experiment.
* 4 threads were not faster than 1 thread at batch size 1. For a model this
  small, threading overhead cancels the parallel speed-up.
* **Model inference vs recognition delay.** The ~0.5 ms is the time to
  *classify* a window that has already been recorded. Recording the window
  takes 2,560 ms. With the dataset's 50 % overlap, a new window completes
  every 1,280 ms, and each one covers the preceding 2.56 s. After an activity
  change, the first window made entirely of the new activity completes
  2.56–3.84 s later (window length plus up to one hop), although a classifier
  may respond earlier to partially filled windows. End-to-end recognition
  delay is therefore on the order of seconds and set by the windowing scheme. Inference adds about 0.02 % to the
  2,560 ms window (about 0.04 % of the 1,280 ms hop). Shortening the window is
  the lever for responsiveness; a faster model is not. Sensor I/O, buffering
  and app scheduling were not measured.
* For LogReg, scikit-learn's `predict` (~0.05–0.07 ms) is about 7–10× slower
  than the arithmetic it performs (~0.007–0.008 ms in NumPy). The baseline's
  measured latency is mostly library overhead.

### 5.4 Robustness to additive sensor noise (additional experiment)

Gaussian noise was added to the standardised **test** windows only. σ is a
fraction of each channel's training-partition SD. In physical units, σ = 1
corresponds to ≈0.35–0.41 g on total_acc, 0.10–0.19 g on body_acc, and
0.25–0.40 rad/s on the gyro. Noise was applied to all channels, to the
accelerometer-derived channels only, or to the gyro channels only. Each cell
is the mean over 5 seeds × 3 noise draws. Source:
[robustness.csv](results/robustness.csv); figure:
[robustness_cnn.png](results/figures/robustness_cnn.png).

| CNN, test macro-F1 | σ = 0 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| **noise on all channels:** acc | 0.902 | 0.905 | 0.897 | 0.715 | 0.506 |
| gyro | 0.788 | 0.659 | 0.595 | 0.303 | 0.212 |
| acc+gyro | 0.912 | 0.912 | 0.907 | 0.809 | 0.592 |
| total_acc | 0.891 | 0.882 | 0.601 | 0.454 | 0.273 |
| total_acc+gyro | 0.912 | 0.909 | 0.874 | 0.617 | 0.482 |
| **noise on gyro only:** acc+gyro | 0.912 | 0.911 | 0.911 | 0.907 | **0.781** |
| total_acc+gyro | 0.912 | 0.911 | 0.898 | **0.750** | **0.577** |

Findings:

* Under noise on all channels, the fused model's *mean* macro-F1 degrades
  more slowly than either single-sensor model. Seed variance is large at σ ≥
  0.5 (SD ≈ 0.10 for acc and acc+gyro at σ = 0.5; see `robustness.csv`).
* **When only the gyroscope degrades, fusion becomes a liability.** At σ = 1
  on the gyro, acc+gyro (0.781) falls below the *clean* acc-only model
  (0.902). For the 6-channel control, the crossover comes earlier: at σ = 0.5,
  total_acc+gyro (0.750) is below clean total_acc (0.891). The fused network
  relies on gyro features and has never seen a degraded gyro. Sensor-dropout
  or noise augmentation would be the standard fix (§10).
* The gyro-only model is the most fragile: −0.13 macro-F1 already at σ = 0.1.
* Caveat: the apparent robustness of the 6-channel `acc` input vs 3-channel
  `total_acc` is partly an **artefact**. Noise was drawn independently for
  total_acc and its derived copy body_acc, so the network effectively gets two
  noisy views of one signal. A real accelerometer's noise would be shared by
  both.
* LogReg is almost unaffected by noise (acc+gyro 0.529 → 0.503 at σ = 1).
  Under strong L2 regularisation it averages over 1,152 inputs, which washes
  out i.i.d. noise. It is robust but uninformative.

## 6. Supplementary sensitivity analyses (post-hoc)

Both were added after inspecting the main run. They use the same protocol (no
test-based selection) and live in separate result folders.

**A. LogReg C at the grid boundary** (`configs/supplementary/logreg_extended_C.yaml`).
The main-run validation sweep chose C = 0.001, the smallest value in the grid,
for 4 of 5 conditions. With the grid extended to {1e-5 … 1e-2}, validation now
peaks at an interior value (C = 3e-4 or 1e-3). Test macro-F1 changed by at
most 0.025:
* acc: 0.504 → 0.478
* gyro: 0.292 → 0.271
* total_acc+gyro: 0.515 → 0.503
* acc+gyro (0.529) and total_acc (0.450): unchanged

The baseline's level is therefore not an artefact of the grid. But the LogReg
fusion-vs-acc gap doubled (+0.025 → +0.050) purely through re-selection of C,
so small LogReg differences are fragile.

**B. A different validation split** (`configs/supplementary/val_split_seed1.yaml`;
validation subjects {1, 15, 16, 23, 29} instead of {8, 11, 16, 19, 23}; CNN, 5
seeds). Test macro-F1:
* acc: 0.903 ± 0.007
* gyro: 0.790 ± 0.014
* acc+gyro: 0.916 ± 0.008
* total_acc: 0.890 ± 0.025
* total_acc+gyro: 0.916 ± 0.005

Fusion vs acc: +0.0129 ± 0.0046 seed Δ (5/5 seeds), subject Δ +0.016 [+0.0015,
+0.0373], 7/9 subjects, Wilcoxon p = 0.098, Holm p = 0.29. Fusion vs gyro:
+0.126 (Holm p = 0.059). This is the same qualitative picture as the main run:
a small, direction-consistent fusion gain over the accelerometer that is not
significant at the subject level.

## 7. Failure analysis

* **Sitting ↔ standing is the dominant error for every accelerometer
  model.** Summed over seeds, 21 % of sitting windows are predicted as standing
  by the acc CNN and 19 % by the acc+gyro CNN. 11–12 % of standing windows go
  the other way. With a waist-mounted phone, the two postures differ mainly in
  a small trunk-tilt difference, and the gyro (rotation rate ≈ 0 when
  stationary) adds little.
* **Errors are concentrated in a few subjects.** Per-subject macro-F1 for the
  acc+gyro CNN (mean over seeds) is ≥0.866 for 7 of 9 test subjects, but 0.742
  for S9 and 0.591 for S10 ([per_subject_cnn.png](results/figures/per_subject_cnn.png),
  [subject_confusions.json](results/subject_confusions.json)).
  * S2: 166 of 230 sitting windows (summed over 5 seeds) predicted standing.
    The subject-specific posture is indistinguishable to the model.
  * S9: the acc-only CNN called 151 of 245 upstairs windows "walking". The gyro
    fixes most of this (24 of 245 with fusion). This one subject supplies most
    of the fusion gain (+0.106 macro-F1 for S9, between −0.035 and +0.024 for
    every other subject). Standing is still mostly predicted as sitting (164
    of 225).
  * S10: 108 of 290 laying windows were predicted as **walking upstairs**, a
    physically implausible confusion. That suggests an unusual phone
    orientation or placement for this subject (gravity on an axis the model
    links to stair climbing). Fusion makes S10 slightly worse (−0.035).
* **Validation is a weak selection signal.** Validation macro-F1 is about
  0.91 for all four accelerometer-containing conditions, so validation alone
  could not have ranked them. Early stopping picked epoch 1–3 in 4 of 25 main
  CNN runs (`cnn_acc_gyro_s4` epoch 1, `cnn_acc_gyro_s2` epoch 2, `cnn_acc_s3`
  and `cnn_acc_raw3_s0` epoch 3). These runs still scored 0.88–0.91 on test,
  so the effect on accuracy is small. But the "best
  checkpoint" partly reflects fluctuations across the 5 validation subjects
  rather than convergence.

## 8. Methodology review (self-audit)

**What was decided before vs after seeing test results.**

| item | when | effect on reported numbers |
|---|---|---|
| Research question, hypotheses, metrics, sensor conditions incl. channel-count controls, model architectures, all hyperparameters, seeds, validation split, latency protocol, noise levels | before the main run (`configs/default.yaml`) | defines all main results (§5) |
| Latency moved to a separate post-training stage | after the smoke test, before the main run | protocol only |
| Supplementary A: extended LogReg C grid | post-hoc, triggered by the *validation* sweep hitting the grid edge | reported separately (§6A); main table unchanged |
| Supplementary B: second validation split | post-hoc, after seeing main results | reported separately (§6B); main table unchanged |
| Holm correction | post-hoc | makes claims *more* conservative |
| Per-subject confusion analysis (§7) | post-hoc, exploratory | descriptive only; no model changed |

No model, hyperparameter or main-table number was changed after the test set
was evaluated.

**Automated audit.** `scripts/audit_results.py` ran 352 checks on the main run
and 0 failed; the two supplementary runs also passed with 0 failures. Logs:
`results/audit_log.txt` and `results/supplementary/audit_*.txt`. A second
script, `scripts/verify_report_claims.py`, recomputes the numbers quoted in
this report and the README from `results/`, and checks that the exact strings
appear in the documents. It also re-derives every Wilcoxon p-value by exact
enumeration. The audit checks:
* subject disjointness for every run;
* validation subjects drawn only from the official training subjects;
* scaler = training-partition statistics, and ≠ train+val statistics;
* accuracy and macro-F1 recomputed from the stored confusion matrix;
* stored test macro-F1 reproduced by reloading every model;
* LogReg: lbfgs converged for every C, and the selected C is the validation argmax;
* best epoch = argmax of the stored validation curve, and stopping exactly
  `patience` epochs later;
* `summary.csv` recomputed from `runs.csv`;
* bit-identical retraining of one run.

**Metric correctness.**
* Macro-F1 is unweighted over the 6 classes (`average="macro"`).
* Per-subject F1 uses all 6 labels, and every test subject has all 6.
* Latency summaries are means over seeds of per-run medians and p95s; they
  are not percentiles of a mean.

**Claims this study does *not* support**, despite how they might be read:
* "Sensor fusion significantly improves HAR accuracy." It does vs gyro-only.
  Vs accelerometer-only, the subject-level evidence is inconclusive (§5.2,
  §6B).
* "The fused model is slower." Not measurable at the ~±0.1 ms resolution here.
* "Sub-millisecond inference on a phone / embedded device." The measurement
  was on a laptop x86 CPU. Mobile/ARM latency was not measured.
* "The model is robust to sensor noise." Only synthetic i.i.d. Gaussian noise
  was tested, at test time, and the fused model fails under gyro-only
  degradation.
* Any statement about subjects beyond the 9 test subjects of one dataset, or
  about other sensor placements.

**Deviations and blockers.**
* *Environment blocker:* on this machine, Windows Application Control
  blocked the newest scikit-learn wheel (1.9.1): `ImportError: DLL load failed
  while importing _openmp_helpers: An Application Control policy has blocked
  this file.` The policy was **not** bypassed. The study instead uses the
  system's already-installed, allowed packages (scikit-learn 1.7.2, torch
  2.12.0+cpu, NumPy 2.3.2, SciPy 1.16.3), pinned in `requirements.txt`. No
  experiment was skipped because of this.
* The GPU in the machine was not used; everything ran on CPU, as intended.
* The latency benchmark was moved out of the training loop after the smoke
  test showed noisy timings when measured right after training. This was a
  protocol fix decided before the main run.

## 9. Limitations

1. **Few evaluation subjects (9).** This is the main limit on every
   comparative claim. Subject variance (0.59–1.00 per-subject macro-F1)
   dwarfs the fusion effect.
2. **One dataset, one placement:** a waist-mounted Samsung Galaxy S II, 30
   adults aged 19–48, in a lab protocol (per the dataset description). Other
   devices, placements, populations and free-living behaviour are untested.
3. **`body_acc` (and the authors' noise filtering) was computed offline by
   the dataset authors**, and the `.names` file does not say whether the
   filters are causal. A streaming system would have to reproduce them
   causally. The `total_acc` (+gyro) controls avoid this issue
   and perform about the same.
4. **No hyperparameter search for the CNN.** The fixed configuration gives
   every condition an equal budget, but per-condition tuning might change
   small gaps.
5. **The validation set is small (5 subjects)**, so early stopping is noisy
   (§7).
6. **Latency was measured on one laptop CPU**, Python/PyTorch eager mode,
   batch 1. There was no quantisation, TorchScript or ONNX, and no device
   measurement. Run-to-run noise (~±0.1 ms) limits resolution.
7. **The noise model is synthetic**: i.i.d. Gaussian, independent across
   derived channels, not calibrated to a real sensor's noise spectrum. Bias,
   drift, saturation, clock skew and axis misalignment were not tested.
8. **The LogReg baseline is intentionally simple** (flattened windows). A
   feature-based baseline (e.g. the dataset's 561 features or per-channel
   statistics) was not run, so "CNN vs classical ML" is not settled here.
   Only "CNN vs flattened linear" is.

## 10. Next experiments (ordered by expected information gain)

1. **Grouped cross-validation over all 30 subjects** (leave-one-subject-out or
   5-fold by subject, nested validation) to get 30 subject-level
   observations instead of 9. This directly addresses limitation 1 and could
   settle H1 vs accelerometer.
2. **Sensor-dropout and noise augmentation during training** (randomly zero or
   corrupt the gyro or accelerometer stream). Test whether the fusion
   liability in §5.4 disappears without hurting clean accuracy.
3. **Orientation robustness:** random 3D rotation augmentation and/or
   orientation-invariant inputs (e.g. magnitude + gravity-aligned frame),
   motivated by the S10 laying→upstairs failure and sit/stand confusions.
4. **Deployment-realistic inputs and cost:** causal filtering only
   (total_acc + gyro), window-length ablation (1.28 s / 2.56 s / 5.12 s; window
   length dominates decision delay), int8 quantisation, and latency measured
   on a target ARM device.
5. **Equal-budget hyperparameter search per condition** on validation, plus a
   stronger classical baseline (statistical features + logistic regression or
   gradient boosting).
6. **Realistic noise models:** a measured IMU noise spectrum, bias and drift,
   and noise shared between total_acc and its derived channels.

## Appendix: reproducing

See [README.md](README.md) for details and the machine-specific Windows
note. Short version (Windows PowerShell, Python 3.13, CPU):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m motionsense all --config configs/default.yaml
.venv\Scripts\python scripts\audit_results.py --config configs/default.yaml --retrain cnn_acc_gyro_s1
.venv\Scripts\python scripts\verify_report_claims.py
```

`requirements-lock.txt` records the exact versions of the full dependency
closure used for these results.

Expected runtime on the study machine: ~14 minutes for the main run.
