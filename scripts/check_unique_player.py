# Load CSV
import pandas as pd

df = pd.read_csv("data/processed/training_oncourt_features.csv")

# Count unique player IDs
unique_players = df["Player_id"].nunique() if "Player_id" in df.columns else None
print(unique_players)