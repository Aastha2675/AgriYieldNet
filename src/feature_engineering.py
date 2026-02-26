import os
import logging
import pandas as pd
import numpy as np
import yaml

# =====================================================
# LOGGER SETUP
# =====================================================

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("feature_engineering")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "feature_engineering.log")
        )

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# =====================================================
# LOAD CONFIG
# =====================================================

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# =====================================================
# FEATURE ENGINEERING FUNCTION
# =====================================================

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")

    # Convert Dates
    df["ACQ_START_DATE"] = pd.to_datetime(df["ACQ_START_DATE"], errors="coerce")
    df["SOWING_DATE"] = pd.to_datetime(df["SOWING_DATE"], errors="coerce")

    # Temporal Features
    df["acq_month"] = df["ACQ_START_DATE"].dt.month
    df["acq_dayofyear"] = df["ACQ_START_DATE"].dt.dayofyear

    df["days_after_sowing"] = (
        df["ACQ_START_DATE"] - df["SOWING_DATE"]
    ).dt.days

    # Sort for time-series features
    df = df.sort_values(["PLOT_ID", "ACQ_START_DATE"])

    # Rolling + Lag NDVI
    df["ndvi_rolling_mean_7"] = (
        df.groupby("PLOT_ID")["NDVI"]
          .rolling(window=7, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    df["ndvi_lag_1"] = df.groupby("PLOT_ID")["NDVI"].shift(1)
    df["ndvi_lag_2"] = df.groupby("PLOT_ID")["NDVI"].shift(2)

    # Seasonal Aggregation (Plot + Year)
    seasonal_features = df.groupby(["PLOT_ID", "YEAR"]).agg({

        "NDVI": ["mean", "max", "min", "std"],
        "EVI": ["mean", "max"],
        "SAVI": ["mean"],
        "GNDVI": ["mean"],
        "NDWI": ["mean"]

    }).reset_index()

    seasonal_features.columns = [
        "PLOT_ID", "YEAR",
        "ndvi_mean", "ndvi_max", "ndvi_min", "ndvi_std",
        "evi_mean", "evi_max",
        "savi_mean",
        "gndvi_mean",
        "ndwi_mean"
    ]

    df = df.merge(seasonal_features,
                  on=["PLOT_ID", "YEAR"],
                  how="left")

    # Interaction Features
    df["ndvi_evi_interaction"] = df["NDVI"] * df["EVI"]
    df["ndvi_savi_interaction"] = df["NDVI"] * df["SAVI"]
    df["ndvi_ndwi_interaction"] = df["NDVI"] * df["NDWI"]

    # Growth stage ratio
    if "DAYS_SOW_TO_HARVEST" in df.columns:
        df["growth_stage_ratio"] = (
            df["days_after_sowing"] / df["DAYS_SOW_TO_HARVEST"]
        )

    # Drop NaNs created by lag
    df = df.dropna()

    logger.info("Feature engineering completed.")
    logger.info(f"Final Shape: {df.shape}")

    return df


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():
    logger.info("Feature Engineering Stage Started")

    config = load_config()

    processed_dir = config["paths"]["project_processed_dir"]

    train_input_path = os.path.join(
        processed_dir, config["files"]["train_output"]
    )

    test_input_path = os.path.join(
        processed_dir, config["files"]["test_output"]
    )

    train_output_path = os.path.join(
        processed_dir, config["files"]["train_fe_output"]
    )

    test_output_path = os.path.join(
        processed_dir, config["files"]["test_fe_output"]
    )

    # Load data
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    # Apply feature engineering
    train_fe = apply_feature_engineering(train_df)
    test_fe = apply_feature_engineering(test_df)

    # Save
    train_fe.to_csv(train_output_path, index=False)
    test_fe.to_csv(test_output_path, index=False)

    logger.info("Feature engineered datasets saved successfully.")


if __name__ == "__main__":
    main()