import h5py
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import re

# ── ユーザー設定 ─────────────────────
TARGET_LAT = 34.2
TARGET_LON = 136.7
DATA_FOLDER = "your_folder"  # 適宜パス変更
SLOPE = 0.0012
OFFSET = -10
# ────────────────────────────────────

def find_nearest_pixel(lat_array, lon_array, target_lat, target_lon):
    """指定座標に最も近いピクセルのインデックスを返す"""
    dist = (lat_array - target_lat)**2 + (lon_array - target_lon)**2
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return i, j

def extract_sst_from_file(file_path, target_lat, target_lon):
    """指定ファイルから対象地点のSSTを取得し、物理量に変換"""
    with h5py.File(file_path, 'r') as f:
        sst_dn = f['Image_data']['SST_AVE'][:]

    rows, cols = sst_dn.shape
    lat = np.linspace(90, -90, rows)
    lon = np.linspace(0, 360, cols)
    lon2d, lat2d = np.meshgrid(lon, lat)

    i, j = find_nearest_pixel(lat2d, lon2d, target_lat, target_lon)
    dn = sst_dn[i, j]

    if dn in [65535, -9999]:
        return np.nan
    return dn * SLOPE + OFFSET

# ── ファイル処理 ─────────────────────
file_list = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.h5")))

dates_asc, sst_asc = [], []
dates_desc, sst_desc = [], []

for file_path in file_list:
    basename = os.path.basename(file_path)

    # 日付抽出
    match = re.search(r"(\d{8})", basename)
    if not match:
        continue
    date_str = match.group(1)
    date = pd.to_datetime(date_str, format="%Y%m%d")

    # 軌道タイプ判定（8桁日付の直後の文字）
    idx = basename.find(date_str)
    if idx == -1 or len(basename) <= idx + 8:
        continue
    orbit_type = basename[idx + 8]  # 'A' または 'D'

    # SST抽出
    sst = extract_sst_from_file(file_path, TARGET_LAT, TARGET_LON)

    if orbit_type == "A":
        dates_asc.append(date)
        sst_asc.append(sst)
    elif orbit_type == "D":
        dates_desc.append(date)
        sst_desc.append(sst)

# ── DataFrame作成 ─────────────────────
df_asc = pd.DataFrame({"Date": dates_asc, "SST": sst_asc}).sort_values("Date").set_index("Date")
df_desc = pd.DataFrame({"Date": dates_desc, "SST": sst_desc}).sort_values("Date").set_index("Date")

# ── プロット ─────────────────────
# NaNを除去（dropna）して線が途切れないようにする
df_asc_clean = df_asc.dropna()
df_desc_clean = df_desc.dropna()

# プロット（線が途切れません）
plt.figure(figsize=(12, 5))
plt.plot(df_asc_clean.index, df_asc_clean["SST"], marker='o', linestyle='-', label="Ascending")
plt.plot(df_desc_clean.index, df_desc_clean["SST"], marker='s', linestyle='-', label="Descending", color='orange')
plt.title(f"SST at ({TARGET_LAT}, {TARGET_LON}) - Ascending vs Descending")
plt.xlabel("Date")
plt.ylabel("SST [°C]")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
