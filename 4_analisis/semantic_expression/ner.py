from numpy import e
import pandas as pd 
from collections import Counter, defaultdict
import json

from preprocessing.individual_functions import normalize_ner
def read_json(src):
    with open(src)as f:
        data=json.load(f)
    return data 

def clean_entities(data):
    cleaned=[]
    valid={"GPE", "LOC", "FAC", "ORG"}
    
    for row in data:
        index=row["indice"]
        for entity in row["entities"]:
            if entity["label"] not in valid:
                continue 
            if len(entity["text"])<3:
                continue
            cleaned.append({
                "index":index,
                "text":normalize_ner(entity["text"]),
                "label":entity["label"]
            })
    return cleaned
def aggregate_entities(cleaned):
    entity_map=defaultdict(lambda:{
        "count":0,
        "indices":set()
    })
    for item in cleaned:
        key=(item["text"],item["label"])
        entity_map[key]["count"] +=1
        entity_map[key]["indices"].add(item["index"])
    result=[]
    for (text,label),val in entity_map.items():
        result.append({
            "text":text,
            "label":label,
            "count":val["count"],
            "indices":list(val["indices"])
        })
    result.sort(key=lambda x :x["count"],reverse=True)
    return result
