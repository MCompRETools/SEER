!pip install sentence-transformers pandas scikit-learn
import pandas as pd

# Load Excel
df = pd.read_excel("Train Dataset.xlsx")

# Rename for clarity (optional)
df = df.rename(columns={
    df.columns[0]: 'req1',
    df.columns[1]: 'req2',
    df.columns[2]: 'label'
})
df = df.dropna(subset=['req1', 'req2', 'label'])
# Check class balance
print(df['label'].value_counts())
positive_df = df[df.label == 1]
negative_df = df[df.label == 0]
positive_sample = positive_df.sample(n=len(negative_df), random_state=42)
balanced_df = pd.concat([positive_sample, negative_df]).sample(frac=1, random_state=42).reset_index(drop=True)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Convert training pairs into InputExamples
train_examples = [
    InputExample(texts=[row.req1, row.req2], label=int(row.label))
    for _, row in train_df.iterrows()
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True
)

# Save the model
model.save('./fine_tuned_req_model_balanced')

from sentence_transformers.util import cos_sim
from sklearn.metrics import classification_report

# Reload model (optional)
model = SentenceTransformer('./fine_tuned_req_model_balanced')

preds, labels = [], []

for _, row in test_df.iterrows():
    emb1 = model.encode(row.req1, convert_to_tensor=True)
    emb2 = model.encode(row.req2, convert_to_tensor=True)
    sim = cos_sim(emb1, emb2).item()

    # Threshold: 0.5 (tuneable)
    preds.append(1 if sim > 0.5 else 0)
    labels.append(int(row.label))

# Classification metrics
report = classification_report(labels, preds, target_names=["Unrelated", "Related"])
print(report)