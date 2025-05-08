import io
import pandas as pd
from docx import Document
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
class DocumentProcessor:
    def __init__(self):
        cfg = yaml.safe_load(open("config/settings.yaml"))["processing"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"]
        )

    def process(self, data, filename):
        text = self._extract_text(data, filename)
        chunks = self.splitter.split_text(text)
        return [
            {"text": ch, "metadata": {"source": filename, "chunk_id": f"{filename}-{i}"}}
            for i, ch in enumerate(chunks)
        ]

    def _extract_text(self, data, filename):
        ext = filename.lower().split('.')[-1]
        if ext == "pdf":
            return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)
        if ext == "docx":
            return "\n".join(para.text for para in Document(io.BytesIO(data)).paragraphs)
        if ext == "csv":
            return pd.read_csv(io.BytesIO(data)).to_string()
        if ext == "txt":
            return data.decode()
        raise ValueError(f"Unsupported file: {filename}")
