#!pip install datasets
from datasets import load_dataset
ds = load_dataset("Skylion007/openwebtext") #Training Dataset loading 
dolly = load_dataset("databricks/databricks-dolly-15k", split="train") #Fine Tuning dataset loading

