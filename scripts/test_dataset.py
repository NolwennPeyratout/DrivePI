import json

path = "/home/renault/repo/dataset/nuscenes/LiDAR-LLM-Nu-Caption/train_plain_llava.json"
with open(path) as f:
    data = json.load(f)

print("Total items loaded:", len(data))
for i, item in enumerate(data):
    conv = item.get("conversations")
    if not isinstance(conv, list) or len(conv) != 2:
        print("Enregistrement invalide:", i, type(conv).__name__,
              len(conv) if isinstance(conv, list) else None)