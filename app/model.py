import torch
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')
mlb = joblib.load('models/mlb.pkl')
nn_model = torch.load('models/nn_model.pth', map_location='cpu')
nn_model.eval()

def predict(text: str) -> dict:
    embedding = embedder.encode([text])
    X = torch.tensor(embedding, dtype=torch.float32)

    with torch.no_grad():
        reg_out, clf_out = nn_model(X)

    esg_score = float(reg_out[0][0])
    clf_probs = clf_out[0].numpy()
    clf_binary = (clf_probs > 0.5).astype(int)
    sdg_labels = mlb.inverse_transform(clf_binary.reshape(1, -1))[0]

    return {
        "esg_score": round(esg_score, 2),
        "sdg_labels": list(sdg_labels),
        "sdg_probabilities": {
            label: round(float(prob), 3)
            for label, prob in zip(mlb.classes_, clf_probs)
        }
    }