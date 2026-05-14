from structura.synthetic import build_catalog, generate_dataset
from structura.validators import validate_sample_record


def test_synthetic_dataset_records_are_schema_valid_and_grounded() -> None:
    catalog = build_catalog(seed=7, per_category=3)
    records = generate_dataset(catalog, num_samples=40, seed=7)

    assert len(records) == 40
    assert {record["scenario"] for record in records}

    invalid = [validate_sample_record(record) for record in records]
    assert all(not errors for errors in invalid)
