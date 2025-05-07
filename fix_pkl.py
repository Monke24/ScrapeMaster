import pickle

with open("tag_encoder.pkl", "rb") as f:
    mlb = pickle.load(f)

print("Classes:", mlb.classes_)
print("Type:", type(mlb))