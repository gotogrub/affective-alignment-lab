from collections import Counter

from datasets import load_dataset

from affective_lab.ekman import EKMAN_LABELS, labels_to_multihot, format_emotion_vector


def main() -> None:
    dataset = load_dataset("AiLab-IMCS-UL/go_emotions-en")

    counter = Counter()
    multilabel_count = 0

    for row in dataset["train"]:
        label_ids = row["labels_ekman"]

        if len(label_ids) > 1:
            multilabel_count += 1

        for label_id in label_ids:
            counter[label_id] += 1

    print("Ekman label distribution in train split:")
    print()

    total = sum(counter.values())

    for label_id, count in sorted(counter.items()):
        label_name = EKMAN_LABELS[label_id]
        percent = count / total * 100
        print(f"{label_id}: {label_name:10s} {count:6d}  {percent:6.2f}%")

    print()
    print(f"Rows with multiple Ekman labels: {multilabel_count}")

    print()
    print("Example multihot vector:")
    sample = dataset["train"][0]
    vector = labels_to_multihot(sample["labels_ekman"])

    print(sample["text"])
    print(format_emotion_vector(vector))


if __name__ == "__main__":
    main()