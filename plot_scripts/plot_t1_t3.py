import re
import pandas as pd
import argparse
import os

HAMMER_ROOT = os.environ['HAMMER_ROOT']
bitflip_pattern = re.compile(
    r"Observed 1 bit-flip\(s\) in Row (\d+), Byte (\d+), Address (0x[0-9a-f]+)\s+"
    r"The (\d+)[a-z]{2} bit flipped from (\d) to (\d) \(Data Pattern: 0x([0-9a-fA-F]+) -> 0x([0-9a-fA-F]+)\)\s+"
)

parser = argparse.ArgumentParser()

parser.add_argument('--banks', nargs='+', required=True,
                    help="List of bank IDs to run the campaign on.")
parser.add_argument('--num_aggs', type=int, nargs='+', default=[24],
                    help="List of number of aggressors to run the campaign on.")
parser.add_argument('--campaign_dir', type=str, 
                    default=os.path.join(HAMMER_ROOT, 'results', 'campaign'),
                    help="Path to the directory containing the campaign results.")

args = parser.parse_args()

seen = set()
bitflips = []

for bank_id in args.banks:
    target_dir = os.path.join(args.campaign_dir, f'bank_{bank_id}')

    for dirpath, dirnames, filenames in os.walk(target_dir):
        for num_aggs in args.num_aggs:
            for filename in filenames:
                if not 'log' in filename:
                    continue
                if not f'{num_aggs}agg' in filename:
                    continue
                full_path = os.path.join(dirpath, filename)
                print(f'Collecting results from {filename}...')

                with open(full_path, "r") as f:
                    log_data = f.read()

                for match in bitflip_pattern.finditer(log_data):
                    row, byte, addr, bit, from_bit, to_bit, original, observed = match.groups()
                    key = (bank_id, int(row), num_aggs)
                    if key not in seen:
                        seen.add(key)
                        bitflips.append({
                            "Bank": bank_id,
                            "Row": int(row),
                            "Victim Pattern": f"0x{original}",
                            "Observed Pattern": f"0x{observed}",
                            "Flip Direction": f"{from_bit} -> {to_bit}",
                            "Byte Location": int(byte),
                            "Bit Location": int(bit),
                            "Address": addr,
                            "Num Aggressors": num_aggs
                        })

# Convert to DataFrame
df = pd.DataFrame(bitflips)

t1_df = df
# Plot Table 1 ...
if not df.empty:
    t1_df = (
        df.groupby(['Bank', 'Num Aggressors'])['Row']
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=args.num_aggs, fill_value=0)
    )

# Create formatted table with multi-line header and aligned vertical bars
lines = []
aggs = list(t1_df.columns)
banks = list(t1_df.index)

# Determine column widths
col_width = 4
header_title = "n-Sided Patterns"
bank_col = "Bank"
header_line1 = f"{bank_col:<{col_width}} | {header_title:^{(len(aggs) * 5 - 2) if len(aggs) != 0 else 0}}"
header_line2 = f"{'':<{col_width}} | " + " ".join(f"{agg:^5}" for agg in aggs)

lines.append(header_line1)
lines.append(header_line2)
lines.append("-" * (col_width + 3 + len(aggs) * 6))

for bank in banks:
    row = f"{bank:<{col_width}} | " + " ".join(f"{t1_df.loc[bank, agg]:^5}" for agg in aggs)
    lines.append(row)

# Save table as a text file
t1_output_path = os.path.join(args.campaign_dir, "t1.txt")
with open(t1_output_path, "w") as f:
    f.write("\n".join(lines))



# Plot Table 3 ...
t3_df = df.drop_duplicates(subset=["Bank", "Row"])
lines = []
separator = "-" * 80
lines.append(separator)

lines.append(
    f"{'Bank':<6}"
    f"{'Row':<7}"
    f"{'Victim':<10}"
    f"{'Observed':<10}"
    f"{'Flip':<11}"
    f"{'Byte':<10}"
    f"{'Bit':<10}"
    f"{'Address'}"
)
lines.append(
    f"{'':<13}"
    f"{'Pattern':<10}"
    f"{'Pattern':<10}"
    f"{'Direction':<11}"
    f"{'Location':<10}"
    f"{'Location':<10}"
    f"{''}"
)

lines.append(separator)

for _, row in t3_df.iterrows():
    lines.append(
        f"{row['Bank']:<6}"
        f"{row['Row']:<7}"
        f"{row['Victim Pattern']:<10}"
        f"{row['Observed Pattern']:<10}"
        f"{row['Flip Direction']:<11}"
        f"{row['Byte Location']:<10}"
        f"{row['Bit Location']:<10}"
        f"{row['Address']}"
    )

lines.append(separator)

# Write to a text file
t3_output_path = os.path.join(args.campaign_dir, "t3.txt")
with open(t3_output_path, "w") as f:
    f.write("\n".join(lines))



