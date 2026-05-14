from structura.dataset import split_group_key, split_records
from structura.synthetic import build_catalog, generate_dataset
from structura.validators import validate_sample_record


def test_synthetic_dataset_records_are_schema_valid_and_grounded() -> None:
    catalog = build_catalog(seed=7, per_category=3)
    records = generate_dataset(catalog, num_samples=120, seed=7)

    assert len(records) == 120
    assert {record["scenario"] for record in records}
    assert any(record["target"]["intent"] == "product_search" for record in records)

    invalid = [validate_sample_record(record) for record in records]
    assert all(not errors for errors in invalid)


def test_grouped_split_prevents_cross_split_duplicate_leakage() -> None:
    catalog = build_catalog(seed=9, per_category=3)
    records = generate_dataset(catalog, num_samples=160, seed=9)
    splits = split_records(records, train_ratio=0.8, valid_ratio=0.1, seed=9)

    seen: dict[str, str] = {}
    for split_name, split_records_ in splits.items():
        assert split_records_
        for record in split_records_:
            key = split_group_key(record, group_by="input_target")
            assert key not in seen or seen[key] == split_name
            seen[key] = split_name
