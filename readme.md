
# API

## Install

### Venv creation

>  python3 -m venv ./.venv/

### Venv activation

#### Windows

> TODO

#### Linux

> chmod +x .venv/bin/activate

> .venv/bin/activate

### Libraries instalation

> pip install -r requirement.txt

## APP Launch

> uvicorn main:app --reload

The API should be accecible at this address+

>  http://127.0.0.1:8000

## .env format

``` shell
APP_NAME="IASearchAPI_default"
NEO4J_URI="NEO4J BDD URI"
NEO4J_USERNAME="username"
NEO4J_PASSWORD="password"
```
