import numpy as np
import pandas as pd
from faker import Faker
import os

RNG = np.random.default_rng(42)  
fake = Faker("en_IN")
Faker.seed(42)

N_PROJECTS = 5000
N_DISTRICTS = 50

STAGES = ["notification", "approval", "compensation", "legal_dispute", "possession", "rehabilitation"]

PROJECT_TYPES = ["Highway", "Railway", "Irrigation", "Urban", "Industrial", "Renewable Energy"]
PROJECT_TYPE_WEIGHTS = [0.28, 0.18, 0.14, 0.16, 0.14, 0.10]

DISPUTE_BASE_PROB = {
    "Highway": 0.35, "Industrial": 0.35, "Railway": 0.25,
    "Urban": 0.30, "Irrigation": 0.15, "Renewable Energy": 0.20,
}

STATES = [
    "Uttar Pradesh", "Maharashtra", "Rajasthan", "Madhya Pradesh", "Bihar",
    "Gujarat", "Karnataka", "Tamil Nadu", "Odisha", "West Bengal",
    "Telangana", "Andhra Pradesh", "Punjab", "Haryana", "Chhattisgarh",
]


def scaled(x):
    """min-max scale an array to 0-1, safe against zero-range."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def clip(x, lo, hi):
    return np.clip(x, lo, hi)



def build_districts(n=N_DISTRICTS):
    rows = []
    for _ in range(n):
        state = RNG.choice(STATES)
        district_name = fake.city()
        historical_delay_rate = RNG.beta(2, 5)  
        rows.append({"state": state, "district": district_name,
                     "district_historical_delay_rate": historical_delay_rate})
    return pd.DataFrame(rows)



def generate_projects(n=N_PROJECTS):
    districts = build_districts()
    district_idx = RNG.integers(0, len(districts), size=n)
    dist_rows = districts.iloc[district_idx].reset_index(drop=True)

    project_type = RNG.choice(PROJECT_TYPES, size=n, p=PROJECT_TYPE_WEIGHTS)

    land_area_hectares = RNG.lognormal(mean=np.log(15), sigma=0.9, size=n)
    land_area_hectares = clip(land_area_hectares, 0.5, 500)

    base_density = 8
    noise_factor = RNG.uniform(0.7, 1.3, size=n)
    num_affected_families = RNG.poisson(lam=np.maximum(land_area_hectares * base_density * noise_factor, 0.1))

    num_approvals_pending = RNG.poisson(lam=2, size=n)
    num_approvals_pending = clip(num_approvals_pending, 0, 8)

    compensation_disbursed_pct = RNG.beta(2, 2, size=n) * 100

    base_prob = np.array([DISPUTE_BASE_PROB[pt] for pt in project_type])
    adjusted_prob = clip(base_prob + 0.15 * scaled(num_affected_families), 0, 0.8)
    legal_dispute_flag = RNG.binomial(1, adjusted_prob)

    base = compensation_disbursed_pct * 0.8
    penalty = np.where(legal_dispute_flag == 1, 25, 0)
    rehab_noise = RNG.beta(2, 3, size=n) * 20 - 10  
    rehab_progress_pct = clip(base - penalty + rehab_noise, 0, 100)

    possession_status = np.where(
        compensation_disbursed_pct < 40, "Not Started",
        np.where((compensation_disbursed_pct < 85) | (legal_dispute_flag == 1), "Partial", "Complete")
    )
    flip_mask = RNG.random(n) < 0.05
    flip_choices = RNG.choice(["Not Started", "Partial", "Complete"], size=n)
    possession_status = np.where(flip_mask, flip_choices, possession_status)

    stakeholder_responsiveness_score = clip(RNG.normal(3, 0.8, size=n), 1, 5)

    days_since_notification = RNG.exponential(scale=300, size=n).astype(int)
    days_since_notification = clip(days_since_notification, 0, 1500)

    df = pd.DataFrame({
        "project_id": [f"PRJ{i:05d}" for i in range(n)],
        "project_type": project_type,
        "state": dist_rows["state"],
        "district": dist_rows["district"],
        "district_historical_delay_rate": dist_rows["district_historical_delay_rate"],
        "land_area_hectares": land_area_hectares.round(2),
        "num_affected_families": num_affected_families,
        "num_approvals_pending": num_approvals_pending,
        "compensation_disbursed_pct": compensation_disbursed_pct.round(1),
        "legal_dispute_flag": legal_dispute_flag,
        "possession_status": possession_status,
        "rehab_progress_pct": rehab_progress_pct.round(1),
        "stakeholder_responsiveness_score": stakeholder_responsiveness_score.round(1),
        "days_since_notification": days_since_notification,
    })
    return df



STAGE_WEIGHTS = {
    "notification":   {"num_approvals_pending": 0.45, "district_historical_delay_rate": 0.30, "stakeholder_responsiveness_score_inv": 0.25},
    "approval":       {"num_approvals_pending": 0.35, "stakeholder_responsiveness_score_inv": 0.35, "district_historical_delay_rate": 0.30},
    "compensation":   {"compensation_disbursed_pct_inv": 0.45, "district_historical_delay_rate": 0.25, "land_area_hectares": 0.30},
    "legal_dispute":  {"legal_dispute_flag": 0.55, "num_affected_families": 0.20, "district_historical_delay_rate": 0.25},
    "possession":     {"legal_dispute_flag": 0.35, "compensation_disbursed_pct_inv": 0.30, "num_affected_families": 0.15, "district_historical_delay_rate": 0.20},
    "rehabilitation": {"num_affected_families": 0.40, "rehab_progress_pct_inv": 0.35, "district_historical_delay_rate": 0.25},
}


def compute_stage_labels(df, percentile_threshold=75):
    out = df.copy()

    feat = {
        "num_approvals_pending": scaled(df["num_approvals_pending"]),
        "district_historical_delay_rate": scaled(df["district_historical_delay_rate"]),
        "stakeholder_responsiveness_score_inv": 1 - scaled(df["stakeholder_responsiveness_score"]),
        "compensation_disbursed_pct_inv": 1 - scaled(df["compensation_disbursed_pct"]),
        "land_area_hectares": scaled(df["land_area_hectares"]),
        "legal_dispute_flag": df["legal_dispute_flag"].astype(float),
        "num_affected_families": scaled(df["num_affected_families"]),
        "rehab_progress_pct_inv": 1 - scaled(df["rehab_progress_pct"]),
    }

    n = len(df)
    for stage, weights in STAGE_WEIGHTS.items():
        risk_score = np.zeros(n)
        for key, w in weights.items():
            risk_score += w * feat[key]
        noise = RNG.normal(0, 0.05, size=n)  
        risk_score = risk_score + noise

       
        shape, base_scale = 8.0, 20.0
        duration = RNG.gamma(shape, base_scale * (1 + 4 * risk_score), size=n)
        duration = clip(duration, 1, None)

        threshold = np.percentile(duration, percentile_threshold)
        delayed = (duration > threshold).astype(int)

        out[f"{stage}_duration_days"] = duration.round(0).astype(int)
        out[f"{stage}_delayed"] = delayed
        out[f"{stage}_risk_score"] = risk_score.round(3)

    return out



def inject_missing(df):
    df = df.copy()
    for col, rate in [("compensation_disbursed_pct", 0.08), ("stakeholder_responsiveness_score", 0.08)]:
        mask = RNG.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def validate(df):
    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nMISSINGNESS:\n", df.isna().mean().round(3)[df.isna().mean() > 0])

    print("\nCLASS BALANCE (delayed %) PER STAGE:")
    for stage in STAGES:
        col = f"{stage}_delayed"
        if col in df.columns:
            print(f"  {stage:15s}: {df[col].mean() * 100:.1f}% delayed")

    print("\nKEY CORRELATION CHECKS (should be clearly non-zero):")
    checks = [
        ("land_area_hectares", "num_affected_families"),
        ("compensation_disbursed_pct", "rehab_progress_pct"),
        ("legal_dispute_flag", "possession_delayed"),
        ("district_historical_delay_rate", "notification_delayed"),
    ]
    for a, b in checks:
        corr = df[a].corr(df[b])
        print(f"  {a} vs {b}: r = {corr:.3f}")

    print("\nSAMPLE ROWS:")
    print(df.head(3).T)


if __name__ == "__main__":
    projects = generate_projects(N_PROJECTS)
    full = compute_stage_labels(projects)
    full = inject_missing(full)

    validate(full)

    out_path = "land_acquisition_projects.csv"
full.to_csv(out_path, index=False)
print(f"\nSaved {len(full)} rows -> {out_path}")
    