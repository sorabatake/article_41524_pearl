# google colabで作業する場合
from google.colab import drive
drive.mount('/content/drive')
# 使用するライブラリをインポート
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import cv2
from collections import Counter
from scipy import stats
# ── ユーザー設定 ───────────────────────────────────
TARGET_LAT   = 34.2
TARGET_LON   = 136.7
DATA_FOLDER  = "your_folder"
SLOPE        = 0.0012
OFFSET       = -10
# ────────────────────────────────────────────
def interpolate_with_cv2(coarse:np.ndarray, interval:int)-> np.ndarray:
    """
    OpenCVを使用して、解像度を補間する関数。
    今回の使い方：GCOM-Cの粗い格子状のSSTデータをの滑らかな状態にする
    参考URL(https://gportal.jaxa.jp/gpr/assets/mng_upload/GCOM-C/GCOM-C_Products_Users_Guide_entrylevel_jp.pdf)p.120
    Args:
        coarse (np.ndarray): 元の解像度のデータ。
        interval (int): 補間の間隔。
    Returns:
        np.ndarray: 補間された解像度のデータ。
    """
    array = np.array(coarse, dtype=np.float32)
    rows, cols = array.shape
    new_size = ((cols - 1) * interval + 1, (rows - 1) * interval + 1)
    resized = cv2.resize(array, new_size, interpolation=cv2.INTER_LINEAR)
    return resized
#データの補間が正しいのか検証
def verify_interpolation_bounds(coarse_array, fine_array, interval, label=""):
    """
    coarse_array: 補間前の配列（2D numpy array）
    fine_array:   cv2.resize による補間後の配列
    interval:     補間倍率（10など）
    label:        緯度・経度などのラベル
    """
    rows, cols = coarse_array.shape
    fine_rows, fine_cols = fine_array.shape
    # 想定される位置
    i_fine_max = (rows - 1) * interval
    j_fine_max = (cols - 1) * interval
    corners = [
        ("左上",    (0, 0),                   (0, 0)),
        ("右上",    (0, cols - 1),            (0, j_fine_max)),
        ("左下",    (rows - 1, 0),            (i_fine_max, 0)),
        ("右下",    (rows - 1, cols - 1),     (i_fine_max, j_fine_max))
    ]
    print(f"\n=== {label} 補間境界の検証 ===")
    for name, (i_coarse, j_coarse), (i_fine, j_fine) in corners:
        v_coarse = coarse_array[i_coarse, j_coarse]
        v_fine   = fine_array[i_fine, j_fine]
        ok = np.isclose(v_coarse, v_fine, atol=1e-4)
        #print(f"{name}: coarse = {v_coarse:.6f}, fine = {v_fine:.6f} → {'OK' if ok else 'NG'}")
def interpolate_and_verify(lat_coarse, lon_coarse, lat_int, lon_int, sst_dn)-> tuple[int, int]:
    """
    緯度・経度の補間と検証を行う関数。
    Args:
        lat_coarse (np.ndarray): 補間前の緯度データ。
        lon_coarse (np.ndarray): 補間前の経度データ。
        lat_int (int): 緯度の補間間隔。
        lon_int (int): 経度の補間間隔。
        sst_dn (np.ndarray): SSTのDN値データ。
    Returns:
        tuple [int, int]: 補間後の緯度・経度のインデックス。
    """
    # 緯度・経度間隔の補完
    full_lat = interpolate_with_cv2(lat_coarse, lat_int)
    full_lon = interpolate_with_cv2(lon_coarse, lon_int)
    rows_full, cols_full = sst_dn.shape
    full_lat = full_lat[:rows_full, :cols_full]
    full_lon = full_lon[:rows_full, :cols_full]
    dists = (full_lat - TARGET_LAT)**2 + (full_lon - TARGET_LON)**2
    i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)
    print(f"  補間位置: ({i_min}, {j_min})")
    # # 検証
    # verify_interpolation_bounds(lat_coarse, full_lat, lat_int, label="緯度")
    # verify_interpolation_bounds(lon_coarse, full_lon, lon_int, label="経度")
    return i_min, j_min
def analyze_error(block: np.ndarray) -> str:
    """
    エラーブロックを解析し、最も頻出するエラータイプを返す関数。
    Args:
        block (np.ndarray): エラーデータを内包するブロック。
    Returns:
        str: 最も頻出するエラータイプのラベル。
    """
    error_types = {
        65535: "error_DN",
        65533: "Cloud_mask_DN",
        65532: "Retrieval_error_DN",
        0: "Missing_data_DN"
    }
    flat = block[~np.isnan(block)].astype(int).flatten()
    mode_val, _ = stats.mode(flat, keepdims=False)
    label = error_types.get(int(mode_val), f"Valid_or_Unknown : {mode_val}")
    return label
def get_dn(sst_dnblock: np.ndarray) -> tuple[float, str]:
    # SST変換
    if sst_dnblock.size == 0:
        print("block is empty")
        return None, "out of range"
    # dn値が65500以上の値は欠損値として扱う(参照URL: https://suzaku.eorc.jaxa.jp/GCOM_C/data/update/Algorithm_SST_ja.html)
    sst_dnblock_with_nan = np.where(sst_dnblock < 65500, sst_dnblock, np.nan)
    try:
        dn_val = int(np.nanmedian(sst_dnblock_with_nan))
        sst_val = dn_val * SLOPE + OFFSET  # DN値から実際の温度に変換(参考URL: https://suzaku.eorc.jaxa.jp/GCOM_C/data/update/Algorithm_SST_ja.html)
        print(f"  SST: DN={dn_val}, 実値={sst_val:.2f}°C")
        if not (-2 <= sst_val <= 40):
            print("  → SST out of realistic range, skip")
            return None, "error_SST"
        else:
            return sst_val, f"Valid_SST : {sst_val:.2f}°C"
    except ValueError as e:
        message = str(e)
        if "cannot convert float NaN to integer" in message:
            # print("  → NaN detected in SST block, analyzing error")
            return None, analyze_error(sst_dnblock)

def get_sst_data(fp: str):
    with h5py.File(fp, "r") as f:
        # h5pyファイルから必要なデータを取得
        m = re.search(r"GC1SG1_(\d{8})", fp)
        if not m:
            print("  → Date parse failed, skip")
            return None, None
        date = pd.to_datetime(m.group(1), format="%Y%m%d")
        # Geometryデータからそれぞれ取得
        lat_coarse = f["Geometry_data/Latitude"][:]  # 緯度の補間前データ
        lon_coarse = f["Geometry_data/Longitude"][:] # 経度の補間前データ
        lat_int = int(f["Geometry_data/Latitude"].attrs["Resampling_interval"][0])  # 緯度の補間間隔
        lon_int = int(f["Geometry_data/Longitude"].attrs["Resampling_interval"][0])  # 経度の補間間隔
        # obs_time_coarse = f["Geometry_data/Obs_time"][:]  # 観測時刻の補間前データ
        dt_from_fp = re.search(r"GC1SG1_(\d{12})", fp)
        obs_time_coarse = f["Geometry_data/Obs_time"][:]
        obs_dt = pd.to_datetime(dt_from_fp.group(1), format="%Y%m%d%H%M")
        sst_dn = f["Image_data/SST"][:]  # SSTのDN値
        cloud_prob = f["Image_data/Cloud_probability"][:]  # 雲確率
        i_min, j_min = interpolate_and_verify(lat_coarse, lon_coarse, lat_int, lon_int, sst_dn)
        kernel = 10
        sst_dnblock = sst_dn[i_min - kernel : i_min + kernel, j_min - kernel : j_min + kernel]
        sst_val, label = get_dn(sst_dnblock)
        # 軌道の取得
        orbit = "Descending" if 6 <= obs_dt.hour <= 18 else "Ascending"
        record = {
            "datetime": obs_dt,
            "SST": sst_val,
            "CloudProb": float(cloud_prob[i_min, j_min]),
            "Orbit": orbit,
            "label": label
        }
        return record
def plot_sst_data(records):
    df = pd.DataFrame([d for d in records if "Valid_SST" in d["label"]])
    if df.empty:
        print("有効なデータが取得できませんでした。")
        quit()
    df = df.sort_values("datetime")
    start = pd.to_datetime("2025-02-01")
    end = pd.to_datetime("2025-02-28")
    df_month = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
    if df_month.empty:
        print("指定期間内に有効データがありません。")
        quit()
    plt.figure(figsize=(12, 6))
    for label, group in df_month.groupby("Orbit"):
        plt.plot(group["datetime"], group["SST"], marker="o",linestyle="--" if label == "Descending" else "-", label=f"{label} Pass")
    plt.xlabel("DateTime")
    plt.ylabel("Sea Surface Temperature (°C)")
    plt.title(f"GCOM-C SST at ({TARGET_LAT}, {TARGET_LON})")
    plt.minorticks_on()
    # datelabels = [date.strftime("%Y-%m-%d") for date in pd.date_range(start, end, freq="D")]
    # plt.set_xticklabels(datelabels, rotation=45)
    plt.xticks(rotation=45)
    plt.grid(which="major", linestyle="-", linewidth=0.8)
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
def main(path: str):
    paths = sorted(glob.glob(path + "*.h5"))
    if not paths:
        print(f"No .h5 files found in {path}")
        quit()
    records, labels = [], []
    for fp in paths:
        print(f"Processing {fp} ...")
        try:
            record = get_sst_data(fp)
            records.append(record)
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    plot_sst_data(records)
main(DATA_FOLDER)