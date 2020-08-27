from pydantic.main import ModelMetaclass
from pydantic import BaseModel, conint, validator, confloat, ValidationError
import pandas as pd
import numpy as np
import typing as t


class PassengerSchema(BaseModel):
    passenger_id: conint(ge=1)
    Survived: conint(ge=0, le=1)
    p_class: t.Optional[int]
    name: t.Optional[str]
    sex: t.Optional[str]
    age: t.Optional[conint(ge=0)]
    sib_sp: t.Optional[conint(ge=0)]
    parch: t.Optional[conint(ge=0)]
    ticket: t.Optional[str]
    fare: t.Optional[confloat(ge=0, le=600)]
    cabin: t.Optional[str]
    embarked: t.Optional[str]

    @validator('sex')
    def sex_validator(cls, value):
        if value is not None and value not in ['male', 'female']:
            raise ValueError('Invalid sex value')
        return value

    @validator('embarked')
    def embarked_validator(cls, value):
        if value is not None and value not in ['S', 'C', 'Q']:
            raise ValueError('Invalid embarked value')
        return value


class DataTemplate:
    def __init__(self, schema: ModelMetaclass):
        self.schema = schema

    def __repr__(self):
        return f'DataTemplate({self.schema().__dict__})'

    def __str__(self):
        return str(self.schema)

    def record(self, record: t.Optional[t.Dict]):
        if record is None:
            record = {}
        return self.schema(**record).dict()

    def records(self, records: t.List[t.Dict]):
        return [self.record(x) for x in records]

    def dataframe(self, records: t.List[t.Dict]) -> pd.DataFrame:
        return pd.DataFrame(self.records(records))


def validate_inputs(data: pd.DataFrame) -> t.Tuple[pd.DataFrame, t.Optional[t.Dict]]:
    json_data = data.replace({np.nan: None}).to_dict(orient='records')
    schema = DataTemplate(PassengerSchema)
    ret_value = {
        'errors': None,
        'validated_inputs': None
    }
    try:
        ret_value['validated_inputs'] = schema.dataframe(json_data).replace({None: np.nan})
    except ValidationError as exc:
        ret_value['errors'] = exc.json()

    return ret_value


def validate_record(record: t.Dict):
    schema = DataTemplate(PassengerSchema)
    schema.record(record)

