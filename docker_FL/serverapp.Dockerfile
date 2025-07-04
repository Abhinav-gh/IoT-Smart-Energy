FROM flwr/serverapp:1.17.0

WORKDIR /app

COPY App/pyproject.toml .

RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flwr-serverapp"]
