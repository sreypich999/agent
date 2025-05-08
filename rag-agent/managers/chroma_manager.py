import yaml
import chromadb
from sentence_transformers import SentenceTransformer

class ChromaManager:
    def __init__(self):
        cfg = yaml.safe_load(open("config/settings.yaml"))["chroma"]
        self.client = chromadb.PersistentClient(path=cfg["persist_directory"])
        self.collection = self.client.get_or_create_collection(
            name=cfg["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_documents(self, docs):
        texts = [d["text"] for d in docs]
        embs = self.embedder.encode(texts).tolist()
        metas = [d["metadata"] for d in docs]
        ids = [m["chunk_id"] for m in metas]
        self.collection.add(documents=texts, embeddings=embs, metadatas=metas, ids=ids)

    def query(self, question):
        q_emb = self.embedder.encode([question]).tolist()[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {"text": d, "source": m["source"], "score": 1 - dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
