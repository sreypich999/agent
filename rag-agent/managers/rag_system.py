# managers/rag_system.py
import re
from managers.minio_handler import MinIOHandler
from managers.document_processor import DocumentProcessor
from managers.chroma_manager import ChromaManager
from managers.deepseek_adapter import DeepSeekAdapter
from managers.history_manager import HistoryManager

class RAGSystem:
    def __init__(self):
        self.minio    = MinIOHandler()
        self.processor= DocumentProcessor()
        self.chroma   = ChromaManager()
        self.deepseek = DeepSeekAdapter()
        self.history  = HistoryManager()

        # 1) Preload all existing docs at startup:
        for obj in self.minio.list_all_objects():
            try:
                data = self.minio.get_object_data(obj.object_name)
                self._process_doc(data, obj.object_name)
            except Exception as e:
                print(f"[startup] error processing {obj.object_name}: {e}")

        # 2) Then continue watching for new uploads:
        self.minio.watch_bucket(self._process_doc)

    def _process_doc(self, data, filename):
        try:
            docs = self.processor.process(data, filename)
            self.chroma.add_documents(docs)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def ask(self, question):
        context = self.chroma.query(question)
        if not context:
            return "No relevant documents found", []

        prompt = self._build_prompt(question, context)
        response = self.deepseek.generate_response(prompt)
        answer, citations = self._parse_response(response)

        if self._validate(answer, context):
            self.history.add_entry(question, answer, [c["source"] for c in context])
            return answer, citations
        return "I cannot answer that based on available documents", []

    # ... _build_prompt, _parse_response, _validate unchanged ...

    def _build_prompt(self, question, context):
        # Build the context section for the prompt
        sources = "\n".join(
            [f"Source: {c['source']} (Score: {c['score']:.2f})\n{c['text']}" for c in context]
        )
        return f"""
You are an AI assistant with access to specific documents. Answer the following question based on the provided context.

Question: {question}

Context:
{sources}

Answer:
"""

    def _parse_response(self, response):
        # Assuming the response is in the format: "Answer: <answer> Sources: <sources>"
        match = re.match(r"Answer: (.*?) Sources: (.*)", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            citations = [s.strip() for s in match.group(2).split(',')]
            return answer, citations
        return response.strip(), []

    def _validate(self, answer, context):
        # Implement validation logic here
        # For example, check if the answer references the provided context
        return any(c['source'] in answer for c in context)
