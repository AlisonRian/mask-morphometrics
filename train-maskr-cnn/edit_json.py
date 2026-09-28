import json

path = r"train-maskr-cnn/dataset/valid"

# Carrega a anotação
with open(f"{path}/_annotations.coco.json", "r") as f:
    data = json.load(f)

data["categories"] = [
    c for c in data["categories"] if c["name"] in ["aerial", "root"]
]
for c in data["categories"]:
    c['supercategory'] = ''

cat_id_map = {cat["id"]: i for i, cat in enumerate(data["categories"])}
for cat in data["categories"]:
    cat["id"] = cat_id_map[cat["id"]]

for ann in data["annotations"]:
    ann["category_id"] = cat_id_map[ann["category_id"]]

# Salva o arquivo corrigido
with open(f"{path}/_annotations_fixed.coco.json", "w") as f:
    json.dump(data, f)