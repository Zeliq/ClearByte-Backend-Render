import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('backend/Database.csv')

# Split data (adjust test_size as needed)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save splits
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
