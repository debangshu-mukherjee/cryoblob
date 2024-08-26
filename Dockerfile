FROM python:3.10-slim AS production

COPY . .
RUN python -m pip install .

FROM production AS development
RUN python -m pip install .[dev]
