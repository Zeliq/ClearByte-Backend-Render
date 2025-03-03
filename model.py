import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score

# Configuration
CONFIG = {
    "model_name": "bert-base-multilingual-cased",
    "data_path": "Database.json",
    "categories": ["vegetarian", "vegan", "halal"],  # Removed 'kosher'
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 15,
    "max_length": 64,
    "dropout_rate": 0.3,
    "threshold": 0.6,
    "few_shot_examples": [
        {"ingredient": "Beef", "vegetarian": 0, "vegan": 0, "halal": 1},
        {"ingredient": "Tofu", "vegetarian": 1, "vegan": 1, "halal": 1},
        {"ingredient": "Shrimp", "vegetarian": 0, "vegan": 0, "halal": 0},
        {"ingredient": "Cheese", "vegetarian": 1, "vegan": 0, "halal": 1},
    ]
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# 1. Load and Prepare Data --------------------------------------
def load_data():
    with open(CONFIG["data_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ingredients = {}
    for category, items in data.items():
        if category not in CONFIG["categories"]:  # Ignore removed categories
            continue
        for ingredient in items:
            if ingredient not in ingredients:
                ingredients[ingredient] = {c: 0 for c in CONFIG["categories"]}
            ingredients[ingredient][category] = 1  # Assign label

    for ingredient in data.get("non_category", []):
        if ingredient not in ingredients:
            ingredients[ingredient] = {c: 0 for c in CONFIG["categories"]}

    # Convert to DataFrame
    df = pd.DataFrame([{"ingredient": k, **v} for k, v in ingredients.items()])

    # Ensure labels exist (fill missing values with 0)
    for category in CONFIG["categories"]:
        df[category] = df[category].fillna(0).astype(int)

    # Add few-shot examples
    few_shot_df = pd.DataFrame(CONFIG["few_shot_examples"])
    df = pd.concat([df, few_shot_df], ignore_index=True)

    return df

# 2. Initialize Model and Tokenizer -----------------------------
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=len(CONFIG["categories"]),
    problem_type="multi_label_classification",
    hidden_dropout_prob=CONFIG["dropout_rate"],
    attention_probs_dropout_prob=CONFIG["dropout_rate"]
).to(device)

# 3. Tokenization -----------------------------------------------
def tokenize_function(batch):
    tokenized = tokenizer(
        batch["ingredient"],
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_length"]
    )
    tokenized["labels"] = [
        [batch[c][i] for c in CONFIG["categories"]]
        for i in range(len(batch["ingredient"]))
    ]
    return tokenized

# 4. Training Setup ---------------------------------------------
def train_model(train_df):
    dataset = Dataset.from_pandas(train_df)
    tokenized_ds = dataset.map(tokenize_function, batched=True)
    tokenized_ds = tokenized_ds.remove_columns(["ingredient"] + CONFIG["categories"])
    
    # Compute class weights
    class_counts = train_df[CONFIG["categories"]].sum(axis=0).replace(0, 1)
    class_weights = (len(train_df) / (len(CONFIG["categories"]) * class_counts)).tolist()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (torch.sigmoid(torch.tensor(logits)) > CONFIG["threshold"]).int().numpy()
        return {
            "f1_weighted": f1_score(labels, preds, average="weighted")
        }
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        greater_is_better=True,
    )
    
    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels").float().to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        eval_dataset=tokenized_ds,
        compute_metrics=compute_metrics,
    )
    
    print("🚀 Starting training...")
    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("✅ Training complete! Model saved")
    return model

# 5. Few-Shot Testing Function -------------------------------------------
def test_model(model, examples=None):
    if examples is None:
        examples = ["Pork", "Chicken", "Garlic", "Shrimp", "Tofu"]
    
    print("\n🧪 Testing Model:")
    for ingredient in examples:
        inputs = tokenizer(
            ingredient,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["max_length"]
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        results = {
            cat: ("✅" if probs[i] > CONFIG["threshold"] else "❌", f"{probs[i]*100:.1f}%") 
            for i, cat in enumerate(CONFIG["categories"])
        }
        
        print(f"\n🔍 {ingredient.capitalize()}:")
        for cat, (symbol, conf) in results.items():
            print(f"  - {cat.capitalize():<12} {symbol} ({conf})")

# 6. Main Execution ---------------------------------------------
if __name__ == "__main__":
    try:
        train_df = load_data()
        if train_df.empty:
            raise ValueError("No data found in Database.json!")
        
        print(f"📊 Dataset size: {len(train_df)} ingredients")
        print("📊 Class distribution:")
        print(train_df[CONFIG["categories"]].sum())
        
        model = train_model(train_df)
        test_model(model)
    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
