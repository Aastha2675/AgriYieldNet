import os
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta


# ===============================
# SETUP LOGGER 
# ===============================

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_cleaning')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_cleaning.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ===============================
# LOAD CONFIG
# ===============================
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# ===============================
# PRE-PROCESS SATELLITE INDICES
# ===============================

def preprocess_satellite_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the date and time from the metadata of satellite_indices dataset.
    """

    try:
        df = raw_df.copy()

        logger.info("Starting extraction for Date and Time on Satellite_indices dataset")

        # Extract datetime directly
        def parse_datetime(folder_name):
            try:
                first_part = folder_name.split("_")[0]
                return pd.to_datetime(first_part, format="%Y%m%dT%H%M%S", errors="coerce")
            except Exception as e:
                logger.error(f"Error parsing folder: {folder_name} | {e}")
                return pd.NaT

        # Check if 'Folder' column exists
        if "Folder" not in df.columns:
            raise KeyError("'Folder' column not found in dataframe")

        # Create ACQ_DATETIME column
        df["ACQ_DATETIME"] = df["Folder"].apply(parse_datetime)

        # Extract components safely
        df["ACQ_YEAR"] = df["ACQ_DATETIME"].dt.year
        df["ACQ_MONTH"] = df["ACQ_DATETIME"].dt.month
        df["ACQ_DAY"] = df["ACQ_DATETIME"].dt.day
        df["ACQ_TIME"] = df["ACQ_DATETIME"].dt.strftime("%H:%M:%S")

        # Optional: create start & end date
        df["ACQ_START_DATE"] = df["ACQ_DATETIME"].dt.date
        df["ACQ_END_DATE"] = df["ACQ_DATETIME"].dt.date

        # Drop temporary column
        df.drop(columns=["ACQ_DATETIME"], inplace=True)

        logger.info("Date and Time extracted successfully.")

        return df

    except Exception as e:
        logger.error(f"Unexpected error in preprocessing satellite dataframe: {e}")
        raise


# ===============================
# CLEAN ANNOTATIONS
# ===============================

def clean_annotation_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs full data cleaning and preprocessing 
    on annotation dataset.
    """
    try:
        df = raw_df.copy()
        logger.info("Starting annotation data cleaning...")

        # -----------------------------
        # Remove rows where SPLIT is null
        # -----------------------------
        df = df[df["SPLIT"].notna()].copy()

        # -----------------------------
        # Data Type Conversion
        # -----------------------------
        int_columns = [
            "PLOT_ID",
            "YEAR",
            "PADDY_BIN",
            "SOWING_DAY",
            "TRANSPLANTING_DAY",
            "HARVESTING_DAY"
        ]

        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # YEAR conversion (safe)
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").fillna(0).astype(int)

        # -----------------------------
        # Handle Missing Values
        # -----------------------------
        if "VARIETY" in df.columns:
            df["VARIETY"] = df["VARIETY"].fillna("Unknown")

        if "RIVER_PART" in df.columns:
            df["RIVER_PART"] = df["RIVER_PART"].fillna("No Water Source")

        if "PADDY_BIN" in df.columns:
            df["PADDY_BIN"] = df["PADDY_BIN"].replace(2, 0)

        # -----------------------------
        # Mixed Date Format Handling
        # (MM-DD-YYYY and MM/DD/YYYY)
        # -----------------------------
        date_columns = [
            "SOWING_DATE",
            "TRANSPLANTING_DATE",
            "HARVESTING_DATE"
        ]

        for col in date_columns:
            if col in df.columns:
                # Standardize separator
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace("-", "/", regex=False)
                )

                # Convert to datetime
                df[col] = pd.to_datetime(
                    df[col],
                    format="%m/%d/%Y",
                    errors="coerce"
                )

                logger.info(f"{col} missing after parsing: {df[col].isna().sum()}")

        # -----------------------------
        # Season Duration Feature
        # -----------------------------
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        def calculate_season_days(season_str, year):
            try:
                if pd.isna(season_str) or year == 0:
                    return np.nan

                start_month_str, end_month_str = season_str.lower().split('-')
                start_month = month_map[start_month_str]
                end_month = month_map[end_month_str]

                start_date = datetime(year, start_month, 1)

                if end_month < start_month:
                    end_date = datetime(year + 1, end_month, 1) + \
                               relativedelta(months=1) - \
                               relativedelta(days=1)
                else:
                    end_date = datetime(year, end_month, 1) + \
                               relativedelta(months=1) - \
                               relativedelta(days=1)

                return (end_date - start_date).days + 1

            except Exception:
                return np.nan

        if "STANDARD_SEASON" in df.columns:
            df["SEASON_DURATION"] = df.apply(
                lambda row: calculate_season_days(
                    row["STANDARD_SEASON"], row["YEAR"]
                ),
                axis=1
            )

        logger.info("Annotation data cleaning completed successfully.")
        logger.info(f"Final cleaned dataset shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise
    
# ===============================
# MERGE DATAFRAMES
# ===============================

def merge_data(sat_df: pd.DataFrame, ann_df: pd.DataFrame) -> pd.DataFrame:

    """Merges annotation dataframe and satellite dataframe."""

    try:
        # Ensure same datatype before merge
        ann_df["UNIQUE_ID"] = ann_df["UNIQUE_ID"].astype(int)
        sat_df["ID"] = sat_df["ID"].astype(int)

        # Merge
        merged_df = pd.merge(
            ann_df,
            sat_df,
            left_on="UNIQUE_ID",
            right_on="ID",
            how="inner"   
        )
        
        merged_df = merged_df.drop_duplicates()
        logger.info(f"Merging completed. Final Dataset shape: {merged_df.shape}")
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise


# ===============================
# MAIN FUNCTION
# ===============================

def main():
    try:
        logger.info("Starting Data Cleaning Pipeline...")

        # Load config
        config = load_config("config.yaml")
        raw_dir = config["paths"]["project_raw_dir"]
        processed_dir = config["paths"]["project_processed_dir"]

        os.makedirs(processed_dir, exist_ok=True)

        # Load raw datasets
        satellite_file = os.path.join(raw_dir, config["files"]["final_output"])
        annotation_file = os.path.join(raw_dir, config["files"]["annotation_output"])

        logger.info(f"Loading satellite data from: {satellite_file}")
        sat_df = pd.read_csv(satellite_file)

        logger.info(f"Loading annotation data from: {annotation_file}")
        ann_df = pd.read_csv(annotation_file)

        # Preprocess satellite data
        sat_df = preprocess_satellite_dataframe(sat_df)

        # Clean annotations
        ann_df = clean_annotation_dataframe(ann_df)

        # Split annotations into train/val/test
        train_ann = ann_df[ann_df["SPLIT"] == "train"].copy()
        val_ann = ann_df[ann_df["SPLIT"] == "val"].copy()
        test_ann = ann_df[ann_df["SPLIT"] == "test"].copy()

        # Merge datasets
        logger.info("Merging Train data...")
        train_final = merge_data(sat_df, train_ann)
        
        logger.info("Merging Validation data...")
        val_final = merge_data(sat_df, val_ann)
        
        logger.info("Merging Test data...")
        test_final = merge_data(sat_df, test_ann)

        # Save processed datasets
        train_file = os.path.join(processed_dir, config["files"]["train_output"])
        val_file = os.path.join(processed_dir, config["files"]["val_output"])
        test_file = os.path.join(processed_dir, config["files"]["test_output"])

        train_final.to_csv(train_file, index=False)
        val_final.to_csv(val_file, index=False)
        test_final.to_csv(test_file, index=False)

        logger.info("Data Cleaning Completed Successfully!")
        logger.info(f"Train Dataset Shape: {train_final.shape}")
        logger.info(f"Validation Dataset Shape: {val_final.shape}")
        logger.info(f"Test Dataset Shape: {test_final.shape}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()