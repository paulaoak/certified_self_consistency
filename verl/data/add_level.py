from datasets import load_dataset
import json

# 1. Load MATH-500 from Hugging Face
ds = load_dataset("HuggingFaceH4/MATH-500")  

# 2. Build a map from an identifier to level
level_map = {}
for ex in ds["test"]:
    ex_id = ex["unique_id"]
    ex_level = ex.get("level")  
    level_map[ex_id] = ex_level

# 3. Load your JSON file
with open("MATH-TTT/test_old.json", "r") as f:
    data = json.load(f)

# 4. Merge level info 
for item in data:
    my_id = item.get("id")
    if my_id in level_map:
        item["difficulty"] = level_map[my_id]
    else:
        item["difficulty"] = None  # or "unknown"

# 5. Save the enriched JSON
with open("MATH-TTT/test.json", "w") as f:
    json.dump(data, f, indent=2)
