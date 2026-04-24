import hdbscan
from sentence_transformers  import SentenceTransformer
from hdbscan import HDBSCAN 
from sklearn.metrics import cluster
from torch import embedding
from umap import UMAP
from bertopic import BERTopic
class BERTopic_analysis():
    def __init__(self,unsupervised,reduction,embedding,docs,*args,**kwargs):
        if unsupervised is None:
            unsupervised=self.set_model_hdbscan()
        if reduction is None:
            reduction=self.set_model_umap()
        if embedding is None:
            embedding="paraphrase-multilingual-MiniLM-L12-v2"
        if docs is None:
            docs=None
        self.unsupervised=unsupervised
        self.reduction=reduction
        self.embedding=embedding
        self.docs=docs 
    def set_model_umap(self,n_neighbors=30,n_components=8,min_dist=0.0,metric='cosine',**kwargs):
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric 
        )
        return umap_model 
    def set_model_hdbscan(self, min_cluster_size=15,metric='euclidean',cluster_selection_method='eom',prediction_data=True,**kwargs):
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size,metric=metric,cluster_selection_method=cluster_selection_method,prediction_data=prediction_data)
        return hdbscan_model
    def embedding_extraction(self,docs,embedding,device="cuda"):
        if embedding is None:
            embedding=self.embedding
        if self.docs is None and docs is None:
            raise ValueError("NO hay documentos")
        if docs is None:
            docs=self.docs        
        self.docs=docs
        model=SentenceTransformer(embedding,device=device)
        emb=model.encode(docs,batch_size=64,show_progress_bar=True)
        self.embedded=emb 
        return emb
    def fit(self):
        if not hasattr(self,"embedded"):
            self.embedding_extraction(self.docs,self.embedding)
        embeddings=self.embedded 
        topic_model=BERTopic(embedding_model=None,umap_model=self.reduction,hdbscan_model=self.unsupervised,language="multilingual",calculate_probabilities=True)
        topics,probs=topic_model.fit_transform(self.docs,embeddings)
        self.model=topic_model
        self.topics=topics
        self.probs=probs
        return topics,probs
    def get_topics(self):
        return self.model.get_topic_info()
    def get_topic(self, topic_id):
        return self.model.get_topic(topic_id)
    def transform(self, new_docs):
        embeddings = self.embedding_extraction(new_docs,self.embedding)
        return self.model.transform(new_docs, embeddings)
