import torch
from transformers import pipeline


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def print_predictions(predictions: list[dict]) -> None:
    sorted_predictions = sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True,
    )

    print()
    print("Emotion scores:")
    print("-" * 40)

    for item in sorted_predictions:
        label = item["label"]
        score = item["score"]
        print(f"{label:10s}: {score:.4f}")

    print("-" * 40)
    print()


def main() -> None:
    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {'cuda' if device == 0 else 'cpu'}")
    print()

    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        top_k=None,
        device=device,
    )

    print("Type a sentence and press Enter.")
    print("Type 'exit' or 'q' to quit.")
    print()

    while True:
        text = input("> ").strip()

        if text.lower() in {"exit", "q", "quit"}:
            print("Bye.")
            break

        if not text:
            continue

        result = classifier(text)

        # pipeline returns: [[{label, score}, ...]]
        predictions = result[0]
        print_predictions(predictions)


if __name__ == "__main__":
    main()