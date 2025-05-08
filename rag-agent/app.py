
import os
import io
import json
import time
import threading
import logging
import re
import sqlite3
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from minio import Minio
from sentence_transformers import SentenceTransformer
from docx import Document
from PyPDF2 import PdfReader
import pandas as pd
import requests
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Load environment variables ───────────────────────────────────────
load_dotenv()
MINIO_ENDPOINT    = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY  = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY  = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")

# ─── Configuration constants ─────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "document-knowledge")
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 200))

# ─── Logging setup ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"MinIO: {MINIO_ENDPOINT}/{MINIO_BUCKET_NAME}")
logger.info(f"DeepSeek API key set: {bool(DEEPSEEK_API_KEY)}")

# ─── Handlers & Managers ─────────────────────────────────────────────
class MinIOHandler:
    def __init__(self):
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        self.bucket = MINIO_BUCKET_NAME
        self._ensure_bucket()
        self.scan_interval = 30

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def list_all(self):
        return self.client.list_objects(self.bucket, recursive=True)

    def fetch(self, name):
        return self.client.get_object(self.bucket, name).read()

    def watch(self, callback):
        def loop():
            seen = set()
            while True:
                current = {o.object_name for o in self.client.list_objects(self.bucket)}
                for new in current - seen:
                    data = self.fetch(new)
                    callback(data, new)
                seen = current
                time.sleep(self.scan_interval)
        threading.Thread(target=loop, daemon=True).start()

class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def process(self, data, filename):
        ext = filename.lower().rsplit('.', 1)[-1]
        if ext == 'pdf':
            text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)
        elif ext == 'docx':
            text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
        elif ext in ('txt', 'csv'):
            text = data.decode() if ext == 'txt' else pd.read_csv(io.BytesIO(data)).to_string()
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        chunks = self.splitter.split_text(text)
        return [
            {"text": ch, "metadata": {"source": filename, "chunk_id": f"{filename}-{i}"}}
            for i, ch in enumerate(chunks)
        ]

class ChromaManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def add(self, docs):
        texts = [d['text'] for d in docs]
        embs = self.embedder.encode(texts).tolist()
        metas = [d['metadata'] for d in docs]
        ids = [m['chunk_id'] for m in metas]
        self.collection.add(documents=texts, embeddings=embs, metadatas=metas, ids=ids)

    def query(self, query_text):
        q_emb = self.embedder.encode([query_text]).tolist()[0]
        res = self.collection.query(
            query_embeddings=[q_emb], n_results=5,
            include=['documents','metadatas','distances']
        )
        return [
            {"text": doc, "source": meta['source'], "score": 1 - dist}
            for doc, meta, dist in zip(
                res['documents'][0], res['metadatas'][0], res['distances'][0]
            )
        ]

class DeepSeekAdapter:
    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")
        self.url = 'https://api.deepseek.com/v1/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }

    def generate(self, prompt: str) -> str:
        payload = {
            'model': 'DeepSeek-V3-Base',
            'messages': [
                {'role': 'system', 'content': 'You are a document assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1
        }
        resp = requests.post(self.url, json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

class HistoryManager:
    def __init__(self):
        os.makedirs('data', exist_ok=True)
        self.conn = sqlite3.connect('data/history.db', check_same_thread=False)
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                question TEXT,
                answer TEXT,
                sources TEXT
            )'''
        )

    def add(self, question, answer, sources):
        self.conn.execute(
            'INSERT INTO history(timestamp,question,answer,sources) VALUES(?,?,?,?)',
            (datetime.now().isoformat(), question, answer, json.dumps(sources))
        )
        self.conn.commit()

class RAGSystem:
    def __init__(self):
        self.minio = MinIOHandler()
        self.processor = DocumentProcessor()
        self.chroma = ChromaManager()
        self.deepseek = DeepSeekAdapter()
        self.history = HistoryManager()

        # preload existing docs
        for obj in self.minio.list_all():
            data = self.minio.fetch(obj.object_name)
            self._ingest(data, obj.object_name)
        # watch for new
        self.minio.watch(self._ingest)

    def _ingest(self, data, filename):
        docs = self.processor.process(data, filename)
        self.chroma.add(docs)

    def ask(self, question: str):
        context = self.chroma.query(question)
        if not context:
            return 'No relevant documents found.', []

        prompt = self._build_prompt(question, context)
        resp = self.deepseek.generate(prompt)
        answer, citations = self._parse(resp)

        if self._validate(answer, context):
            self.history.add(question, answer, [c['source'] for c in context])
            return answer, citations
        return 'Cannot answer from available documents.', []

    def _build_prompt(self, question, context):
        ctx_str = '\n'.join(
            f"Source: {c['source']} (Score: {c['score']:.2f})\n{c['text']}" 
            for c in context
        )
        return f"You are an AI assistant.\nQuestion: {question}\nContext:\n{ctx_str}\nAnswer:"

    def _parse(self, text):
        m = re.match(r'Answer: (.*?) Sources: (.*)', text, re.DOTALL)
        if m:
            return m.group(1).strip(), [s.strip() for s in m.group(2).split(',')]
        return text.strip(), []

    def _validate(self, ans, context):
        return any(c['source'] in ans for c in context)

# ─── Streamlit UI ──────────────────────────────────────────────────────
def main():
    st.title('Document-Based Q&A System')
    st.write('Ask a question based on documents in MinIO:')
    rag = RAGSystem()
    question = st.text_input('Your Question:')
    if question:
        answer, cites = rag.ask(question)
        st.write('**Answer:**', answer)
        if cites:
            st.write('**Citations:**')
            for c in cites:
                st.write('-', c)

if __name__ == '__main__':
    main()

