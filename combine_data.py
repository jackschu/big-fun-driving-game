import csv


def clean_path(dir_name, file_name):
    results = []
    rows = {}
    with open(dir_name + file_name) as csvfile:
        reader = csv.reader(csvfile)  # change contents to floats
        for row in reader:  # each row is a list
            img_name = row[0]
            if img_name == "filename":
                rows[img_name] = [
                    "width,height,class,xmin,ymin,xmax,ymax,xmin,ymin,xmax,ymax,xmin,ymin,xmax,ymax,xmin,ymin,xmax,ymax,xmin,ymin,xmax,ymax"
                ]
            elif img_name in rows:
                rows[img_name] += row[-4:]
            else:
                rows[img_name] = row[1:]

    with open(dir_name + "cleaned_" + file_name, "w") as file:
        for key in rows.keys():
            file.write(key)
            file.write(",")
            print(",".join(rows[key]), file=file)


clean_path("images/test/", "test_labels.csv")
clean_path("images/train/", "train_labels.csv")
