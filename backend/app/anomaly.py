import torch
from sklearn.ensemble import IsolationForest

def z_score_anomaly_scores(z):
    """
    Compute anomaly scores for each node based on z-score distance from the mean embedding.
    Higher score indicates more deviation (i.e., higher anomaly).
    """
    mean = z.mean(dim=0)
    std = z.std(dim=0) + 1e-6  # avoid divide-by-zero
    scores = ((z - mean) / std).pow(2).sum(dim=1).sqrt()
    return scores

def top_anomalies(z, top_k=10):
    """
    Returns top_k anomalous nodes and their scores.
    """
    scores = z_score_anomaly_scores(z)
    top_vals, top_indices = torch.topk(scores, top_k)
    return top_indices.cpu().tolist(), top_vals.cpu().tolist()


def isolate_anomalies(z, contamination=0.03, n_top=10):
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(z.cpu().numpy())
    scores = iso.decision_function(z.cpu().numpy()) * -1  # higher = more anomalous
    top_indices = scores.argsort()[-n_top:]
    return [(int(i), round(scores[i], 4)) for i in top_indices]