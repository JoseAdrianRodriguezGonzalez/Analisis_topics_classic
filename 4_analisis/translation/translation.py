from torch.nn.modules import padding
from transformers import MarianMTModel,MarianTokenizer
import torch
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule 
from tqdm import tqdm
class Translator:
    def __init__(self,model_name="Helsinki-NLP/opus-mt-en-es") :
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer=MarianTokenizer.from_pretrained(model_name)
        self.model=MarianMTModel.from_pretrained(model_name).to(self.device)
    def translate_batch(self,texts,batch_size=32):
        results=[]
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Traduciendo"):
            batch=texts[i:i+batch_size]
            inputs=self.tokenizer(batch,return_tensors="pt",padding=True,truncation=True).to(self.device)
            outputs=self.model.generate(**inputs)
            decoded=[self.tokenizer.decode(t,skip_special_tokens=True) for t in outputs]
            results.extend(decoded)
        return results
