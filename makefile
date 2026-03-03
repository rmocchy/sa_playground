PYTHON := .direnv/python-3.13.3/bin/python
PIP    := .direnv/python-3.13.3/bin/pip
ST     := .direnv/python-3.13.3/bin/streamlit

.PHONY: run install lint fmt clean

## アプリを起動する
run:
	direnv exec . $(ST) run app.py

## 依存パッケージをインストールする
install:
	direnv exec . $(PIP) install -r requirements.txt

## 構文チェック (ruff がある場合)
lint:
	direnv exec . $(PYTHON) -m ruff check app.py

## フォーマット (ruff がある場合)
fmt:
	direnv exec . $(PYTHON) -m ruff format app.py

## キャッシュを削除する
clean:
	rm -rf __pycache__ .streamlit/cache
