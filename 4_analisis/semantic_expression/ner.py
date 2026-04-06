import pandas as pd 
from collections import Counter, defaultdict
import json
from rapidfuzz import fuzz
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
def merge_similar_entities(entities,threshold=90):
    merged=[]
    used=[False]*len(entities)
    for i,base in enumerate(entities):
        if used[i]:
            continue
        base_text=base["text"]
        new_entity={
            "text":base["text"],
            "label":base["label"],
            "count":base["count"],
            "indices":set(base["indices"])
        }
        for j in range(i+1,len(entities)):
            if used[j]:
                continue
            compare=entities[j]
            comp_text=compare["text"]
            if fuzz.ratio(base_text,comp_text)>=threshold:
                new_entity["count"]+=compare["count"]
                new_entity["indices"].update(compare["indices"])
                used[j]=True
        used[i]=True 
        new_entity["indices"]=list(new_entity["indices"])
        merged.append(new_entity)
    return merged
def enrichment_text(groups,original,top_k=5):
    original_map={
        row["indice"]:row 
        for row in original 
    }
    for i,base in enumerate(groups):
        phrases=[]
        for index in base["indices"]:
            if index not in original_map:
                continue 
            phrases.extend(original_map[index]["noun_phrases"])
        counter=Counter(phrases)
        top_phrases=[p for p,_ in counter.most_common(top_k)]
        groups[i]["top_noun_phrases"]=top_phrases
        groups[i]["noun_phrases_freq"]=dict(counter)
    return groups 
