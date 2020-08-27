FROM python:3.7.5-slim-buster AS production 

COPY requirements.txt $HOME/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt \
    && mkdir project

WORKDIR project

CMD ["python", "hello_world.py"]


FROM python:3.7.5-slim-buster AS test 

COPY requirements.test.txt $HOME/
COPY requirements.txt $HOME/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.test.txt \
    && rm requirements.test.txt requirements.txt \
    && mkdir project

WORKDIR project

CMD ["pytest", "tests/"]

