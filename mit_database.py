import os
import wfdb
import pandas as pd

# ----------- PATH SETTINGS -----------
input_folder = "/home/ajay/Desktop/amish/mit-bih-arrhythmia-database-1.0.0"
output_folder = "/home/ajay/Desktop/amish/csv_file_leadii_mit_information"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# ----------- FUNCTION TO FIND LEAD II -----------
def find_lead_ii(leads):
    """
    Priority:
    1. MLII (standard in MIT-BIH)
    2. Any lead containing 'II'
    3. Any lead containing 'ECG'
    4. Fallback: first channel
    """

    # 1️⃣ Exact match
    if "MLII" in leads:
        return leads.index("MLII")

    # 2️⃣ Any name containing 'II'
    for i, name in enumerate(leads):
        if "II" in name.upper():
            return i

    # 3️⃣ Any ECG-like channel
    for i, name in enumerate(leads):
        if "ECG" in name.upper():
            return i

    # 4️⃣ Fallback: first channel
    print("⚠ Lead II not explicitly found. Using first available channel as fallback.")
    return 0


# ----------- LOOP THROUGH ALL RECORDS -----------
for filename in os.listdir(input_folder):

    # Process only header files
    if not filename.endswith(".hea"):
        continue

    record_name = filename.replace(".hea", "")
    record_path = os.path.join(input_folder, record_name)

    try:
        # ----------- READ .DAT + .HEA TOGETHER -----------
        record = wfdb.rdrecord(record_path)

        # Display available leads
        leads = record.sig_name
        print(f"Processing: {record_name} | Available Leads: {leads}")

        # ----------- FIND LEAD II (ROBUST METHOD) -----------
        lead_index = find_lead_ii(leads)
        selected_lead_name = leads[lead_index]
        print(f"➡ Using lead: {selected_lead_name}")

        # ----------- EXTRACT LEAD II SIGNAL -----------
        lead_ii = record.p_signal[:, lead_index]

        # ----------- CREATE DATAFRAME -----------
        output_df = pd.DataFrame({
            "Sample_Index": range(len(lead_ii)),
            "Lead_II": lead_ii
        })

        # ----------- SAVE TO CSV -----------
        output_path = os.path.join(output_folder, f"{record_name}_LeadII.csv")
        output_df.to_csv(output_path, index=False)

        print(f"✅ Saved: {output_path}\n")

    except Exception as error:
        print(f"❌ Failed to process {record_name}: {error}\n")

print("🎯 All files processed successfully.")


