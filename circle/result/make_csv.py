import os
import csv

folder = os.path.join("result", "Circle")
output_csv = "circle_filenames.csv"

filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for name in filenames:
        writer.writerow([name, "", "", ""])
