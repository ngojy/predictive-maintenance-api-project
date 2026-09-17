import numpy as np
import pandas as pd

def generate_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # machine operating baseline
    rpm = rng.normal(1800, 150, n_samples).clip(1000, 2500)  # revolutions per minute
    load_pct = rng.uniform(20, 100, n_samples)  # load percentage
    hours_since_maintenance = rng.uniform(0, 2000, n_samples)  # hours since last maintenance

    # vibration (mm/s RMS) rises with load, rpm, and wear
    vibration = (
        0.5
        + 0.004 * load_pct
        + 0.0015 * (rpm - 1000)
        + 0.003 * hours_since_maintenance
        + rng.normal(0, 0.2, n_samples)
    ).clip(0, None)  # vibration cannot be negative

    # temperature (°C) rises with load and friction/wear
    temperature = (
        40
        + 0.25  * load_pct
        + 0.01 * hours_since_maintenance
        + 0.5 * vibration
        + rng.normal(0, 3, n_samples)
    )

    # failure risk score combines all factors + nonlinear interaction
    risk_score = (
        0.03 * vibration
        + 0.02 * (temperature - 40)
        + 0.0015 * hours_since_maintenance
        + 0.00002 * vibration * hours_since_maintenance
        + rng.normal(0, 0.3, n_samples)
    )

    threshold = np.quantile(risk_score, 0.85) # 85th percentile as threshold for failure
    failure = (risk_score > threshold).astype(int)

    # create DataFrame
    df = pd.DataFrame({
        'vibration_mm_s': vibration,
        'temperature_C': temperature,
        'rpm': rpm,
        'load_pct': load_pct,
        'hours_since_maintenance': hours_since_maintenance,
        'failure': failure
    })

    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/sensor_data.csv", index=False)
    print(f"Generated {len(df)} rows, failure rate: {df['failure'].mean():.2%}")