import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import zipfile
import os

warnings.filterwarnings('ignore')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FakeNewsDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    def prepare_data(self, df, text_column, label_column):
        # Split the data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_column].values,
            df[label_column].values,
            test_size=0.2,
            random_state=42
        )

        # Create datasets
        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer)

        return train_dataset, val_dataset

    def train_one_epoch(self, train_dataset, val_dataset, batch_size=16, epoch_num=1):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f'Epoch {epoch_num}')
        self.model.train()
        total_train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Average training loss: {avg_train_loss:.4f}')

        # Validation
        self.model.eval()
        val_predictions = []
        val_actual = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions = torch.argmax(outputs.logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_actual.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_actual, val_predictions)
        print(f'Validation Accuracy: {accuracy:.4f}')
        print(classification_report(val_actual, val_predictions))

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            prediction = torch.argmax(outputs.logits, dim=1)

        return "FAKE" if prediction.item() == 1 else "REAL"

def main():
    try:
        zip_path = "news_dataset.zip"
        extract_dir = "extracted_dataset"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted files to {extract_dir}")

        # Search for True.csv and Fake.csv in all subdirectories
        true_path = None
        fake_path = None
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file == "True.csv":
                    true_path = os.path.join(root, file)
                elif file == "Fake.csv":
                    fake_path = os.path.join(root, file)
            if true_path and fake_path:
                break
        if not true_path or not fake_path:
            print("True.csv or Fake.csv not found in the extracted directory.")
            return
        df_true = pd.read_csv(true_path)
        df_true['label'] = 0  # Real news
        df_fake = pd.read_csv(fake_path)
        df_fake['label'] = 1  # Fake news
        df = pd.concat([df_true, df_fake], ignore_index=True)
        print(f"Loaded {len(df_true)} true and {len(df_fake)} fake news articles.")

        # Shuffle the combined dataframe
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print("Columns in dataset:", df.columns.tolist())
        detector = FakeNewsDetector()

        # Split into train, validation, and test sets (70/15/15)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

        # Prepare datasets
        train_dataset = NewsDataset(train_df['text'].values, train_df['label'].values, detector.tokenizer)
        val_dataset = NewsDataset(val_df['text'].values, val_df['label'].values, detector.tokenizer)
        test_dataset = NewsDataset(test_df['text'].values, test_df['label'].values, detector.tokenizer)

        # Train interactively
        num_epochs = 3
        for epoch in range(1, num_epochs + 1):
            detector.train_one_epoch(train_dataset, val_dataset, batch_size=16, epoch_num=epoch)
            if epoch < num_epochs:
                cont = input(f"Run next epoch ({epoch+1})? (y/n): ")
                if cont.lower() != 'y':
                    break

        # Test evaluation
        print("\nEvaluating on test set...")
        test_loader = DataLoader(test_dataset, batch_size=16)
        detector.model.eval()
        test_predictions = []
        test_actual = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(detector.device)
                attention_mask = batch['attention_mask'].to(detector.device)
                labels = batch['labels'].to(detector.device)
                outputs = detector.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                test_predictions.extend(predictions.cpu().numpy())
                test_actual.extend(labels.cpu().numpy())
        accuracy = accuracy_score(test_actual, test_predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(test_actual, test_predictions))

        # Prediction demo
        sample_text = input("Enter a news text to predict (or leave blank to exit): ")
        if sample_text:
            result = detector.predict(sample_text)
            print(f"Prediction: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()