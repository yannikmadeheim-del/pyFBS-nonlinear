python -m venv env 	
.\env\Scripts\activate
pip install .
pip install nbconvert
pip install jupyterlab
pytest -v ./tests/build_examples.py