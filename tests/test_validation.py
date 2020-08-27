from my_model.processing.validation import validate_record
from pydantic import ValidationError
import pytest


def test_validator_detects_incorrect_value_in_record(data_record):
    data_record['sex'] = 'it'

    with pytest.raises(ValidationError) as exc:
        validate_record(data_record)

    assert 'value_error' in str(exc)


def test_validator_detects_missing_field_in_record(data_record):
    del data_record['Survived']

    with pytest.raises(ValidationError) as exc:
        validate_record(data_record)

    assert 'field required' in str(exc)


def test_validator_no_errors_correct_record(data_record):
    validate_record(data_record)

