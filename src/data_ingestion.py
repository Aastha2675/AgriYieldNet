import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling
import yaml


# =====================================================
# LOGGER SETUP
# =====================================================

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("data_ingestion")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# =====================================================
# CONFIG LOADER
# =====================================================

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# =====================================================
# RESAMPLING FUNCTION
# =====================================================

def resample_band(src, target_transform, target_width, target_height):
    data = src.read(1)
    dst = np.empty((target_height, target_width), dtype=data.dtype)

    reproject(
        source=data,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=target_transform,
        dst_crs=src.crs,
        resampling=Resampling.bilinear
    )

    return dst


# =====================================================
# VEGETATION INDEX CALCULATION
# =====================================================

def calculate_indices(blue, green, red, nir, swir):

    eps = 1e-10

    NDVI = (nir - red) / (nir + red + eps)
    GNDVI = (nir - green) / (nir + green + eps)
    SAVI = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    ARVI = (nir - (2*red - blue)) / (nir + (2*red - blue) + eps)
    EVI = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)

    # Safe MSAVI
    msavi_term = (2*nir + 1)**2 - 8*(nir - red)
    msavi_term = np.where(msavi_term < 0, 0, msavi_term)
    MSAVI = (2*nir + 1 - np.sqrt(msavi_term)) / 2

    GCI = (nir / (green + eps)) - 1
    SIPI = (nir - red) / (nir - blue + eps)
    NDWI = (green - swir) / (green + swir + eps)
    CI = (nir - swir) / (nir + swir + eps)

    NDWI2 = (nir - swir) / (nir + swir + eps)
    MSI = swir / (nir + eps)
    GLA = (2*green - red - blue) / (2*green + red + blue + eps)
    WI = (green - blue) / (red - green + eps)
    NGRDI = (green - red) / (green + red + eps)
    RGBVI = (green*green - red*blue) / (green*green + red*blue + eps)
    VARI = (green - red) / (green + red - blue + eps)
    ExR = 1.4 * red - green
    ExG = 2 * green - red - blue
    ExGR = ExG - ExR

    return [
        NDVI, GNDVI, SAVI, ARVI, EVI, MSAVI,
        GCI, SIPI, NDWI, CI,
        NDWI2, MSI, GLA, WI, NGRDI,
        RGBVI, VARI, ExR, ExG, ExGR
    ]


# =====================================================
# MAIN DATA INGESTION
# =====================================================

def data_ingestion(source_dir: str) -> pd.DataFrame:

    logger = setup_logger()
    logger.info("Starting data ingestion")

    results = []

    for root, dirs, files in os.walk(source_dir):

        rel_path = os.path.relpath(root, source_dir)
        if rel_path == ".":
            continue

        parts = rel_path.split(os.sep)
        if len(parts) < 2:
            continue

        ID = parts[0]
        folder_name = parts[1]

        try:
            required = {
                "B02": ".B2.tif",
                "B03": ".B3.tif",
                "B04": ".B4.tif",
                "B08": ".B8.tif",
                "B11": ".B11.tif",
                "SCL": ".SCL.tif"
            }

            band_files = {}

            for band, pattern in required.items():
                match = [f for f in files if pattern in f]
                if match:
                    band_files[band] = os.path.join(root, match[0])

            if len(band_files) < 6:
                logger.warning(f"Missing bands in {ID}/{folder_name}")
                continue

            logger.info(f"Processing {ID}/{folder_name}")

            with rasterio.open(band_files["B02"]) as src_b2, \
                 rasterio.open(band_files["B03"]) as src_b3, \
                 rasterio.open(band_files["B04"]) as src_b4, \
                 rasterio.open(band_files["B08"]) as src_b8, \
                 rasterio.open(band_files["B11"]) as src_b11, \
                 rasterio.open(band_files["SCL"]) as src_scl:

                ref_transform = src_b2.transform
                ref_w, ref_h = src_b2.width, src_b2.height

                B2 = src_b2.read(1).astype("float32")
                B3 = resample_band(src_b3, ref_transform, ref_w, ref_h).astype("float32")
                B4 = resample_band(src_b4, ref_transform, ref_w, ref_h).astype("float32")
                B8 = resample_band(src_b8, ref_transform, ref_w, ref_h).astype("float32")
                B11 = resample_band(src_b11, ref_transform, ref_w, ref_h).astype("float32")
                SCL = resample_band(src_scl, ref_transform, ref_w, ref_h)

                cloud_mask = np.isin(SCL, [3, 7, 8, 9, 10, 11])

                B2 = np.where(cloud_mask, np.nan, B2)
                B3 = np.where(cloud_mask, np.nan, B3)
                B4 = np.where(cloud_mask, np.nan, B4)
                B8 = np.where(cloud_mask, np.nan, B8)
                B11 = np.where(cloud_mask, np.nan, B11)

                indices = calculate_indices(B2, B3, B4, B8, B11)

                row = [ID, folder_name]

                for idx in indices:
                    row.append(np.nanmean(idx))

                results.append(row)

        except Exception as e:
            logger.error(f"Error processing {ID}/{folder_name}: {e}")
            continue

    columns = [
        "ID", "Folder",
        "NDVI","GNDVI","SAVI","ARVI","EVI","MSAVI",
        "GCI","SIPI","NDWI","CI",
        "NDWI2","MSI","GLA","WI","NGRDI",
        "RGBVI","VARI","ExR","ExG","ExGR"
    ]

    df = pd.DataFrame(results, columns=columns)

    logger.info("Data converted into DataFrame successfully")
    return df


# =====================================================
# SAVE DATA
# =====================================================

def save_data(df: pd.DataFrame, output_file: str):
    logger = setup_logger()
    df.to_csv(output_file, index=False)
    logger.info(f"Vegetation indices saved at {output_file}")


# =====================================================
# COPY ANNOTATIONS (NO SHUTIL)
# =====================================================

def save_annotations(annotation_src: str, output_dir: str, output_name: str):

    logger = setup_logger()
    os.makedirs(output_dir, exist_ok=True)

    destination = os.path.join(output_dir, output_name)

    df_ann = pd.read_csv(annotation_src)
    df_ann.to_csv(destination, index=False)

    logger.info("Annotations file copied successfully")


# =====================================================
# MAIN
# =====================================================

def main():
    try:
        config = load_config()
        logger = setup_logger()

        source_dir = config["paths"]["source_tif_dir"]
        output_dir = config["paths"]["project_raw_dir"]

        output_file = os.path.join(output_dir, config["files"]["final_output"])
        annotation_src = config["paths"]["annotation_file"]
        annotation_output = config["files"]["annotation_output"]

        os.makedirs(output_dir, exist_ok=True)

        df = data_ingestion(source_dir)
        save_data(df, output_file)

        save_annotations(annotation_src, output_dir, annotation_output)

        logger.info("Data ingestion completed successfully")

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()