import os
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
input_folder = "/home/ajay/Desktop/amish/csv_file_leadii_mit_information "   # Folder with Lead II CSV files
output_folder = "/home/ajay/Desktop/amish/neurokit_output"
sampling_rate = 360   # MIT-BIH sampling rate
# ----------------------------------------

os.makedirs(output_folder, exist_ok=True)

all_results = []

for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(input_folder, file)
    print(f"\n📂 Processing {file}...")

    # Initialize placeholders
    quality_zhao = None
    hrv_time, hrv_freq = pd.DataFrame(), pd.DataFrame()
    error_note = ""

    try:
        # ---------- Read the CSV ----------
        df = pd.read_csv(file_path)

        if "Lead_II" not in df.columns:
            error_note = "Lead_II column missing"
            print(f"⚠ {file}: Lead_II column not found. Columns:", list(df.columns))
        else:
            # ---------- Take Lead II ----------
            ecg_raw = df["Lead_II"].dropna()
            ecg_raw = pd.to_numeric(ecg_raw, errors='coerce').dropna()

            if len(ecg_raw) < sampling_rate * 5:
                error_note = f"ECG signal too short ({len(ecg_raw)} samples)"
                print(f"⚠ {file}: ECG too short, still logging result.")
            else:
                # ---------- ECG Processing (Filtering, R-peaks, Cleaning) ----------
                signals, info = nk.ecg_process(ecg_raw, sampling_rate=sampling_rate)

                ecg_cleaned = signals["ECG_Clean"]

                # ---------- ECG Quality (Zhao2018 only) ----------
                try:
                    quality_zhao = nk.ecg_quality(ecg_raw, sampling_rate=sampling_rate, method="zhao2018")
                    quality_zhao = float(pd.Series(quality_zhao).mean())
                except Exception as e:
                    error_note += f" | zhao2018 failed: {e}"

                # ---------- HRV Features ----------
                try:
                    hrv_time = nk.hrv_time(info, sampling_rate=sampling_rate)
                    hrv_freq = nk.hrv_frequency(info, sampling_rate=sampling_rate)
                except Exception as e:
                    error_note += f" | HRV calculation failed: {e}"

                # ---------- SAVE CLEANED ECG ONLY ----------
                clean_df = pd.DataFrame({
                    "Sample_Index": range(len(ecg_cleaned)),
                    "Lead_II_Cleaned": ecg_cleaned
                })

                clean_output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_Clean.csv")
                clean_df.to_csv(clean_output_path, index=False)
                print(f"✅ Clean ECG saved: {clean_output_path}")

                # ---------- VISUALIZATION: RAW vs CLEANED ----------
                plt.figure(figsize=(12, 5))
                plt.plot(ecg_raw.values[:3000], label="Raw ECG", alpha=0.6)
                plt.plot(ecg_cleaned.values[:3000], label="Cleaned ECG", linewidth=2)
                plt.title(f"ECG Raw vs Cleaned - {file}")
                plt.xlabel("Samples")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.tight_layout()

                plot_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_Raw_vs_Clean.png")
                plt.savefig(plot_path)
                plt.close()

                print(f"📊 Visualization saved: {plot_path}")

    except Exception as e:
        error_note += f" | File read/process error: {e}"

    # ---------- Combine HRV Results ----------
    combined = pd.concat([hrv_time, hrv_freq], axis=1) if not hrv_time.empty else pd.DataFrame([{}])

    # Extract file ID
    file_id = os.path.splitext(file)[0].strip()
    combined.insert(0, "Record_ID", file_id)

    combined["File_Name"] = file
    combined["ECG_Quality_Zhao2018"] = quality_zhao
    combined["Error_Note"] = error_note if error_note else "Processed successfully"

    all_results.append(combined)

# ---------- SAVE FINAL HRV FILE ----------
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(output_folder, "ECG_HRV_FullLength.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\n✅ All HRV results saved to: {out_path}")
else:
    print("\n⚠ No ECG files processed.")
