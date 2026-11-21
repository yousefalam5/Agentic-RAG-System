"""
Agentic Retrieval-Augmented Generation (Agentic RAG) Engine
Implements an autonomous, tool-using AI Agent for complex multi-hop question answering:
1. Multi-format Document Ingestion (PDF, PPTX, DOCX, TXT, CSV)
2. Semantic Vector Indexing (ChromaDB + Multilingual MiniLM)
3. Specialized Agent Tools:
   - document_search: Vector semantic retrieval
   - calculator: Deterministic mathematical evaluation
   - document_summary: Abstractive document overview
   - comparative_analyzer: Cross-entity side-by-side comparison
   - table_data_analyzer: Structured DataFrame querying & aggregation
4. Agent Planner & Multi-Step Reasoning Loop (Plan -> Action -> Observation -> Synthesis)
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import chromadb
import numpy as np
import pandas as pd
import torch
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".csv", ".txt"}


@dataclass
class AgentConfig:
    data_dir: Path = field(default_factory=lambda: Path("sample_docs"))
    chroma_dir: Path = field(default_factory=lambda: Path("data/chroma_db"))
    collection_name: str = "agentic_knowledge_base"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 4
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.chroma_dir = Path(self.chroma_dir).expanduser().resolve()


class AgenticEngine:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.config.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.config.embedding_model, device=self.device)
        self._ocr_engine = None
        self._gemini_client = None
        self.dataframes: dict[str, pd.DataFrame] = {}

        if self.config.gemini_api_key:
            self.update_gemini_key(self.config.gemini_api_key)

    @property
    def collection(self):
        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @collection.setter
    def collection(self, val):
        pass

    def update_gemini_key(self, api_key: str) -> bool:
        if not api_key:
            self._gemini_client = None
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key.strip())
            self._gemini_client = genai.GenerativeModel(self.config.gemini_model)
            self.config.gemini_api_key = api_key.strip()
            return True
        except Exception:
            self._gemini_client = None
            return False

    @staticmethod
    def _clean_text(value: str) -> str:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", value)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    def _read_file(self, path: Path) -> list[tuple[str, dict[str, Any]]]:
        suffix = path.suffix.lower()
        output: list[tuple[str, dict[str, Any]]] = []

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = self._clean_text(page.extract_text() or "")
                if text:
                    output.append((text, {"page": page_number}))
            if not output:
                # OCR fallback for image-based PDFs
                try:
                    import pypdfium2 as pdfium
                    from rapidocr_onnxruntime import RapidOCR

                    if self._ocr_engine is None:
                        self._ocr_engine = RapidOCR()

                    pdf = pdfium.PdfDocument(str(path))
                    for page_number, page in enumerate(pdf, start=1):
                        image = page.render(scale=1.5).to_pil()
                        result, _ = self._ocr_engine(np.array(image))
                        if result:
                            page_text = "\n".join(item[1] for item in result if item and len(item) > 1 and item[1])
                            page_text = self._clean_text(page_text)
                            if page_text:
                                output.append((page_text, {"page": page_number}))
                except Exception:
                    pass

        elif suffix == ".pptx":
            presentation = Presentation(str(path))
            for slide_number, slide in enumerate(presentation.slides, start=1):
                pieces = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        text = "\n".join(paragraph.text for paragraph in shape.text_frame.paragraphs)
                        if text.strip():
                            pieces.append(text)
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                            if cells:
                                pieces.append(" | ".join(cells))
                slide_text = self._clean_text("\n".join(pieces))
                if slide_text:
                    output.append((slide_text, {"slide": slide_number}))

        elif suffix == ".docx":
            document = Document(str(path))
            pieces = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        pieces.append(" | ".join(cells))
            text = self._clean_text("\n".join(pieces))
            if text:
                output.append((text, {"section": "document"}))

        elif suffix == ".csv":
            try:
                df = pd.read_csv(path)
                self.dataframes[path.name] = df
                text_repr = df.to_string(index=False)
                output.append((text_repr, {"type": "tabular_dataset", "rows": len(df)}))
            except Exception:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    reader = csv.reader(handle)
                    rows = [", ".join(c.strip() for c in r if c.strip()) for r in reader]
                text = self._clean_text("\n".join(r for r in rows if r))
                if text:
                    output.append((text, {"type": "csv_data"}))

        elif suffix == ".txt":
            text = self._clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                output.append((text, {"type": "plain_text"}))

        return output

    def ingest_files(self, file_paths: list[Path] | None = None) -> dict[str, Any]:
        """Ingests specified files or scans data_dir if none provided."""
        paths_to_process = []
        if file_paths:
            paths_to_process = [Path(p) for p in file_paths if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]
        else:
            if self.config.data_dir.exists():
                for root, _, files in os.walk(self.config.data_dir):
                    for file in files:
                        p = Path(root) / file
                        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                            paths_to_process.append(p)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[dict[str, Any]] = []
        warnings: list[str] = []
        processed_files = 0

        for path in paths_to_process:
            filename = path.name
            try:
                segments = self._read_file(path)
            except Exception as exc:
                warnings.append(f"Could not read {filename}: {exc}")
                continue

            if not segments:
                warnings.append(f"No extractable text in {filename}.")
                continue

            processed_files += 1
            for text, meta in segments:
                docs = splitter.split_text(text)
                for idx, doc in enumerate(docs):
                    chunk_id = hashlib.sha1(f"{filename}_{idx}_{doc[:40]}".encode("utf-8")).hexdigest()
                    metadata = {"source": filename, **meta}
                    chunks.append({"id": chunk_id, "text": doc, "metadata": metadata})

        if not chunks:
            return {"files": processed_files, "chunks": 0, "warnings": warnings}

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        try:
            self.client.delete_collection(self.config.collection_name)
        except Exception:
            pass

        new_collection = self.client.create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            new_collection.upsert(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                embeddings=embeddings[start : start + len(batch)].tolist(),
                metadatas=[item["metadata"] for item in batch],
            )

        return {"files": processed_files, "chunks": len(chunks), "warnings": warnings}

    # ==========================================
    # TOOL 1: Semantic Document Retrieval
    # ==========================================
    def tool_document_search(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """Searches the knowledge base for relevant text passages."""
        if self.collection.count() == 0:
            return {"results": [], "summary": "Knowledge base is empty."}

        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        if res["ids"] and res["ids"][0]:
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                similarity = max(0.0, 1.0 - float(dist))
                results.append({"text": doc, "metadata": meta, "similarity": round(similarity, 4)})

        formatted = "\n\n".join(
            f"[{i+1}] Source: {r['metadata'].get('source')} (Similarity: {r['similarity']:.2f})\n{r['text']}"
            for i, r in enumerate(results)
        )
        return {"results": results, "formatted": formatted}

    # ==========================================
    # TOOL 2: Deterministic Calculator
    # ==========================================
    def tool_calculator(self, expression: str) -> dict[str, Any]:
        """Safely evaluates mathematical expressions (e.g., percentages, ratios, growth rates)."""
        clean_expr = expression.strip()
        # Remove dangerous or non-math characters
        allowed_chars = set("0123456789+-*/()., %eE_")
        filtered_expr = "".join(c for c in clean_expr if c in allowed_chars).replace("%", "/100.0").replace(",", "")

        try:
            # Safe evaluation environment
            safe_dict = {
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "pow": pow,
            }
            result = eval(filtered_expr, {"__builtins__": None}, safe_dict)
            if isinstance(result, float):
                result_str = f"{result:.4f}".rstrip("0").rstrip(".")
            else:
                result_str = str(result)
            return {"expression": clean_expr, "result": result, "formatted": f"{clean_expr} = {result_str}"}
        except Exception as exc:
            return {"expression": clean_expr, "error": str(exc), "formatted": f"Calculation error on '{clean_expr}': {exc}"}

    # ==========================================
    # TOOL 3: Document Summarizer
    # ==========================================
    def tool_document_summary(self, filename: str | None = None) -> dict[str, Any]:
        """Generates an executive summary of a document or the entire collection."""
        if self.collection.count() == 0:
            return {"summary": "No documents indexed."}

        all_docs = self.collection.get(include=["documents", "metadatas"])
        matching_texts = []
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            if filename is None or filename.lower() in meta.get("source", "").lower():
                matching_texts.append(doc)

        if not matching_texts:
            return {"summary": f"No document found matching '{filename}'."}

        sample_context = "\n---\n".join(matching_texts[:6])
        if self._gemini_client:
            try:
                prompt = (
                    "Provide a concise executive summary highlighting key concepts and data from this text:\n\n"
                    f"{sample_context}\n\nEXECUTIVE SUMMARY:"
                )
                res = self._gemini_client.generate_content(prompt)
                return {"summary": res.text.strip(), "chunks_analyzed": len(matching_texts)}
            except Exception:
                pass

        # Local abstractive fallback
        first_passages = [t[:180] + "..." for t in matching_texts[:3]]
        summary = f"Summary across {len(matching_texts)} passages:\n" + "\n".join(f"- {p}" for p in first_passages)
        return {"summary": summary, "chunks_analyzed": len(matching_texts)}

    # ==========================================
    # TOOL 4: Comparative Analyzer
    # ==========================================
    def tool_comparative_analyzer(self, entity_a: str, entity_b: str) -> dict[str, Any]:
        """Retrieves and constructs a structured comparison between two entities."""
        res_a = self.tool_document_search(entity_a, top_k=2)
        res_b = self.tool_document_search(entity_b, top_k=2)

        context_a = res_a.get("formatted", "No data for " + entity_a)
        context_b = res_b.get("formatted", "No data for " + entity_b)

        if self._gemini_client:
            try:
                prompt = (
                    f"Compare and contrast '{entity_a}' vs '{entity_b}' based solely on the provided contexts.\n"
                    f"Present key similarities, differences, and quantitative performance specs in a clear comparative table:\n\n"
                    f"CONTEXT FOR {entity_a}:\n{context_a}\n\n"
                    f"CONTEXT FOR {entity_b}:\n{context_b}\n\n"
                    "COMPARATIVE ANALYSIS:"
                )
                res = self._gemini_client.generate_content(prompt)
                return {"comparison": res.text.strip(), "entity_a": entity_a, "entity_b": entity_b}
            except Exception:
                pass

        comparison = (
            f"### Comparative Overview: {entity_a} vs {entity_b}\n\n"
            f"**Key Evidence for {entity_a}:**\n{context_a}\n\n"
            f"**Key Evidence for {entity_b}:**\n{context_b}\n"
        )
        return {"comparison": comparison, "entity_a": entity_a, "entity_b": entity_b}

    # ==========================================
    # TOOL 5: Tabular Data Analyzer
    # ==========================================
    def tool_table_analyzer(self, question: str) -> dict[str, Any]:
        """Analyzes structured CSV tables (e.g. cloud pricing, instance specs, hardware numbers)."""
        # Load sample CSV if not loaded
        if not self.dataframes:
            csv_path = self.config.data_dir / "cloud_providers_pricing_and_performance.csv"
            if csv_path.exists():
                self.dataframes[csv_path.name] = pd.read_csv(csv_path)

        if not self.dataframes:
            return {"error": "No tabular CSV data available in knowledge base."}

        # Select primary dataframe
        table_name = list(self.dataframes.keys())[0]
        df = self.dataframes[table_name]

        q_lower = question.lower()
        insights = []

        # Heuristic query matching on DataFrame
        if "cheapest" in q_lower or "lowest cost" in q_lower or "min" in q_lower:
            if "hourly_cost_usd" in df.columns:
                cheapest_row = df.loc[df["hourly_cost_usd"].idxmin()]
                insights.append(
                    f"Cheapest instance: `{cheapest_row.get('provider')}` {cheapest_row.get('instance_type')} "
                    f"at ${cheapest_row.get('hourly_cost_usd')}/hour (${cheapest_row.get('monthly_cost_usd')}/mo)."
                )
        if "most expensive" in q_lower or "highest cost" in q_lower or "max" in q_lower:
            if "hourly_cost_usd" in df.columns:
                expensive_row = df.loc[df["hourly_cost_usd"].idxmax()]
                insights.append(
                    f"Most expensive instance: `{expensive_row.get('provider')}` {expensive_row.get('instance_type')} "
                    f"at ${expensive_row.get('hourly_cost_usd')}/hour."
                )
        if "average" in q_lower or "avg" in q_lower or "mean" in q_lower:
            if "hourly_cost_usd" in df.columns:
                avg_cost = df["hourly_cost_usd"].mean()
                insights.append(f"Average hourly cost across all cloud instances: ${avg_cost:.2f}/hour.")

        # If LLM is available, perform intelligent pandas query synthesis
        if self._gemini_client:
            try:
                table_preview = df.head(8).to_markdown()
                prompt = (
                    f"You are a structured data analyst. Given the table preview below:\n\n"
                    f"{table_preview}\n\n"
                    f"Answer this specific question using the data: '{question}'\n"
                    "Cite exact numbers, instance types, and column metrics."
                )
                res = self._gemini_client.generate_content(prompt)
                return {"table_used": table_name, "analysis": res.text.strip(), "shape": df.shape}
            except Exception:
                pass

        if not insights:
            insights.append(f"Table '{table_name}' has {len(df)} rows. Columns: {', '.join(df.columns)}")

        return {"table_used": table_name, "analysis": "\n\n".join(insights), "shape": df.shape}

    # ==========================================
    # AGENT PLANNER & EXECUTION LOOP
    # ==========================================
    def plan_query(self, query: str) -> list[dict[str, Any]]:
        """Decomposes the query into planned sub-tasks and tool selections."""
        q_lower = query.lower()
        plan: list[dict[str, Any]] = []

        # Check for comparative intent
        if any(w in q_lower for w in ["compare", "vs", "versus", "difference between", "مقارنة"]):
            match = re.search(r"compare\s+([^,]+?)\s+(?:and|vs|versus)\s+(.+)", q_lower)
            if match:
                entity_a, entity_b = match.group(1).strip(), match.group(2).strip()
            else:
                entity_a, entity_b = "Entity A", "Entity B"
            plan.append({
                "step": 1,
                "tool": "comparative_analyzer",
                "args": {"entity_a": entity_a, "entity_b": entity_b},
                "intent": f"Perform structured comparison between '{entity_a}' and '{entity_b}'.",
            })

        # Check for tabular / numerical analysis intent
        elif any(w in q_lower for w in ["cheapest", "pricing", "cost", "provider", "instance", "table", "sla", "bandwidth"]):
            plan.append({
                "step": 1,
                "tool": "table_data_analyzer",
                "args": {"question": query},
                "intent": "Query structured CSV cloud pricing table for numerical records.",
            })

        # Check for summarization intent
        elif any(w in q_lower for w in ["summarize", "summary", "overview", "ملخص"]):
            plan.append({
                "step": 1,
                "tool": "document_summary",
                "args": {"filename": None},
                "intent": "Generate comprehensive document summary.",
            })

        # Standard / Multi-hop Semantic Retrieval
        else:
            plan.append({
                "step": 1,
                "tool": "document_search",
                "args": {"query": query, "top_k": self.config.top_k},
                "intent": f"Retrieve top semantic passages matching '{query}'.",
            })

        # If question involves calculation, arithmetic, or ratio
        if any(w in q_lower for w in ["calculate", "how much", "ratio", "percentage", "diff", "multiply", "%", "احسب", "كم"]):
            # Extract possible arithmetic expression or schedule calculator step
            numbers = re.findall(r"\d+(?:\.\d+)?", query)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} / {numbers[1]}" if "ratio" in q_lower else f"{numbers[0]} * {numbers[1]}"
            else:
                expr = "100 - 50"
            plan.append({
                "step": len(plan) + 1,
                "tool": "calculator",
                "args": {"expression": expr},
                "intent": f"Compute numerical metric via deterministic calculator: '{expr}'.",
            })

        return plan

    def run_agentic_pipeline(self, query: str) -> dict[str, Any]:
        """
        Executes the full Agentic RAG loop:
        1. Query Planning
        2. Tool Dispatching & Observation Gathering
        3. Multi-Source Synthesis & Grounded Answer Formulation
        """
        start_time = time.time()
        query = query.strip()
        if not query:
            return {"error": "Query cannot be empty."}

        # Step 1: Generate Plan
        plan = self.plan_query(query)

        # Step 2: Execute Plan
        executed_steps: list[dict[str, Any]] = []
        collected_evidence: list[str] = []
        sources: list[dict[str, Any]] = []

        for step in plan:
            tool_name = step["tool"]
            args = step["args"]
            intent = step["intent"]

            observation: Any = None
            if tool_name == "document_search":
                obs = self.tool_document_search(args.get("query", query), top_k=args.get("top_k", 3))
                observation = obs.get("formatted", "No passages found.")
                for r in obs.get("results", []):
                    sources.append({
                        "source": r["metadata"].get("source", "Document"),
                        "similarity": r.get("similarity", 0.0),
                        "page": r["metadata"].get("page") or r["metadata"].get("slide") or "N/A",
                    })

            elif tool_name == "calculator":
                obs = self.tool_calculator(args.get("expression", "0"))
                observation = obs.get("formatted", obs.get("error"))

            elif tool_name == "document_summary":
                obs = self.tool_document_summary(args.get("filename"))
                observation = obs.get("summary", "Summary unavailable.")

            elif tool_name == "comparative_analyzer":
                obs = self.tool_comparative_analyzer(args.get("entity_a", "A"), args.get("entity_b", "B"))
                observation = obs.get("comparison", "Comparison unavailable.")

            elif tool_name == "table_data_analyzer":
                obs = self.tool_table_analyzer(args.get("question", query))
                observation = obs.get("analysis", "Table analysis failed.")

            executed_steps.append({
                "step": step["step"],
                "tool": tool_name,
                "intent": intent,
                "args": args,
                "observation": observation,
            })
            collected_evidence.append(f"[Tool: {tool_name}]\n{observation}")

        # Step 3: Synthesis
        combined_context = "\n\n---\n\n".join(collected_evidence)

        if self._gemini_client:
            try:
                prompt = (
                    "You are an advanced Agentic AI reasoning assistant. Using the gathered tool observations below, "
                    "synthesize a comprehensive, fact-based response to the user's question.\n"
                    "Structure your answer with clear headers, bullet points, and cite specific tools and documents.\n\n"
                    f"USER QUESTION: {query}\n\n"
                    f"COLLECTED EVIDENCE & TOOL OBSERVATIONS:\n{combined_context}\n\n"
                    "FINAL SYNTHESIZED ANSWER:"
                )
                res = self._gemini_client.generate_content(prompt)
                final_answer = res.text.strip()
            except Exception:
                final_answer = "\n\n".join(f"### Results from `{s['tool']}`:\n{s['observation']}" for s in executed_steps)
        else:
            # Deterministic local synthesis
            answer_parts = []
            answer_parts.append(f"### 🤖 Agent Response to: *\"{query}\"*\n")
            for s in executed_steps:
                answer_parts.append(f"#### 🛠️ Executed Tool: `{s['tool']}`\n{s['observation']}\n")
            final_answer = "\n".join(answer_parts)

        elapsed = round(time.time() - start_time, 2)

        return {
            "query": query,
            "plan": plan,
            "steps": executed_steps,
            "final_answer": final_answer,
            "sources": sources,
            "execution_time": elapsed,
        }
