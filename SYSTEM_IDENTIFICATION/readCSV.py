#test read CSV



import csv


with open('/root/FYP/7LINK_SIMS/data/demo_point1', newline='') as f:
    reader = csv.reader(f)

    # reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        print(row)