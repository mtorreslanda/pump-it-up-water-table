# Pump It Up: which water points in Tanzania still work?

Three-class prediction (*functional*, *functional needs repair*, *non functional*) for ~15,000 water
points, from the [DrivenData competition *Pump it Up: Data Mining the Water Table*](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/).

![Leaderboard: best score 0.8304, rank #6](images/leaderboard.png)

**Best public score 0.8304, peaking at #6 on the leaderboard** (October 2026). The reproducible model
scores **0.8280**, and the notebook explains why the gap between those two numbers is noise rather
than skill.

## Approach

| Step | What | Why |
|---|---|---|
| Data quality | Zeros that are really missing values (year 0, longitude 0…) turned into `NaN` plus `*_miss` flags; text normalised; skewed counts log-transformed | Missingness turned out to be one of the most predictive signals |
| Neighbourhood features | How nearby water points look (same aquifer, installer, funder), computed on train + test without labels | +1 point of accuracy, the largest single gain |
| Geometry | Radial distance features, adapted from the 2016 runner-up solution | +0.24 points |
| Models | XGBoost (tuned with Optuna) as the base, a small MLP and a frequency-encoded Random Forest for diversity; CatBoost in later experiments | The weaker MLP helps because it is wrong on *different* water points |
| Blend | Fixed weights 0.70 / 0.20 / 0.10, 20 / 20 / 10 seeds | Weights chosen before looking at the test score |

## Measuring honestly

A 20% holdout (11,880 water points) with an explicit noise floor: at ~80% accuracy the standard error
is ≈0.0036, so any difference below ~0.007 is treated as noise.

The notebook then runs an experiment on the leaderboard itself. Re-training the same pipeline with
other seeds and blend weights moves the public score between 0.8269 and 0.8294, and on five internal
splits the "best" weights change every time, with all differences inside the noise. **Climbing the top
of the leaderboard is mostly selection on the test score**, which is why the reproducible 0.8280 model
is kept as the reference.

## What did not work

The *needs repair* class (7% of the data) stays at ~0.32 recall. Class weights, a two-stage binary
decomposition and an analysis of uncertain predictions all point the same way: with these features
the intermediate state is not separable. It is a data limitation, documented rather than hidden.

## Run it

1. Download `training_set_values.csv`, `training_set_labels.csv` and `test_set_values.csv` from the
   competition page (free account) and put them next to the notebook.
2. `pip install -r requirements.txt`
3. Run `Pump_it_Up.ipynb` top to bottom (~35–40 min, mostly the final model).

The notebook narrative is in Spanish (it was written for the Machine Learning course of the MSc in
Data Science, Big Data & Business Analytics at Universidad Complutense de Madrid).

---
Miguel Ángel Castro Torreslanda · [mtorreslanda.com](https://mtorreslanda.com)
