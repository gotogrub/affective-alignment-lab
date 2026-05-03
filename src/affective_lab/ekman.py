EKMAN_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]


def labels_to_multihot(label_ids: list[int], num_labels: int = len(EKMAN_LABELS)) -> list[float]:
    vector = [0.0] * num_labels

    for label_id in label_ids:
        if 0 <= label_id < num_labels:
            vector[label_id] = 1.0
        else:
            raise ValueError(f"Unknown Ekman label id: {label_id}")

    return vector


def format_emotion_vector(vector: list[float]) -> str:
    lines = []

    for label, value in zip(EKMAN_LABELS, vector):
        lines.append(f"{label:10s}: {value:.3f}")

    return "\n".join(lines)