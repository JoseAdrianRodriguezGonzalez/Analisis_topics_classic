from scipy.sparse import data
from .ner import *
def pipe():
    data=read_json("data/data_spanish/analysis.json")
    cleaned=clean_entities(data)
    aggregate=aggregate_entities(cleaned)
    print(aggregate)
