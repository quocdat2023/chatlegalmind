from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import secrets
from gemini_handler import GeminiHandler, GenerationConfig, Strategy, KeyRotationStrategy
from flask_cors import CORS
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bcrypt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS").split(",")

# FastAPI setup
app = FastAPI()

# Load the embedding model
embedding_model = SentenceTransformer('hiieu/halong_embedding')  # You can change this to another model if needed
# Load FAISS index
loaded_faiss_index = faiss.read_index("source/faiss_index.index")

# Load JSON data
input_path = "source/metadata.json"
with open(input_path, "r", encoding="utf-8") as f:
    chunks_with_metadata = json.load(f)

# Load graph data
with open('source/graph_anle.json', 'r', encoding='utf-8') as f:
    graph_json = json.load(f)

# Search function
def search(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = loaded_faiss_index.search(query_embedding, top_k)
    results = [{"text": chunks_with_metadata[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return results

# Re-rank function
rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)

def rerank(query, search_results):
    scores = []
    for result in search_results:
        text = str(result["text"])
        inputs = tokenizer.encode_plus(query, text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            score = rerank_model(**inputs).logits.item()
        scores.append(score)

    for i, result in enumerate(search_results):
        result["rerank_score"] = scores[i]

    return sorted(search_results, key=lambda x: x["rerank_score"], reverse=True)

def search_with_rerank(query, top_k=3):
    try:
        search_results = search(query, top_k=top_k)
        if not search_results:
            raise ValueError("No search results returned.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        reranked_results = rerank(query, search_results)
        if not reranked_results:
            raise ValueError("No re-ranked results returned.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return reranked_results

# FastAPI endpoint models
class QueryRequest(BaseModel):
    question: str

class CaseData(BaseModel):
    case_number: str

# Generate response using the prompt template
def prompt_template(docs: List[dict], original_query: str) -> str:
    context = "\n".join([str(doc.get('text', {}).get('text', 'Thông tin không có sẵn')) for doc in docs])
    page = "\n".join([str(doc.get('text', {}).get('page', 'Thông tin không có sẵn')) for doc in docs])
    source = "\n".join([str(doc.get('text', {}).get('source', 'Thông tin không có sẵn'))
                    .replace('/content/drive/MyDrive/data_law/', '')
                    .replace('.pdf', '') for doc in docs])

    original_query = f"'{original_query}'"

    response_prompt = f"""
    `Bạn là một chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. Nhiệm vụ của bạn là phân tích và trả lời các câu hỏi về án lệ pháp luật Việt Nam theo hình thức "Án lệ tương đồng và lập luận pháp lý". Bạn phải đảm bảo câu trả lời chính xác, dễ hiểu và tuân thủ các quy định pháp luật hiện hành.
    
    Dưới đây là câu hỏi bạn cần trả lời:
    {original_query}

    **Thông tin tham khảo để trả lời**:
    Nội dung án lệ tìm được: {context}
    Nguồn án lệ: {source}

    **Hướng dẫn trả lời**:
    1. **Cấu trúc câu trả lời**:
    - **Giới thiệu**: Đưa ra án lệ tương đồng với tình huống cần giải quyết.
    - **Nội dung án lệ tìm được**: Mô tả nội dung chính của án lệ liên quan.
    - **Phân tích tình huống**: Phân tích tình huống cụ thể, chỉ ra các yếu tố pháp lý có liên quan.
    - **Lập luận pháp lý**: Giải thích cơ sở pháp lý, viện dẫn điều luật và công văn hướng dẫn (nếu có).
    - **Kết luận**: Xác định tội danh hoặc hướng xử lý phù hợp theo pháp luật.
    """

    return response_prompt

# FastAPI routes
@app.post("/query")
async def query(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Câu hỏi không hợp lệ!")

        docs = search_with_rerank(query=question, top_k=3)
        
        if not docs:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu phù hợp!")

        response_chain = prompt_template(docs, question)

        api_keys = os.getenv('GEMINI_API_KEYS')
        if not api_keys:
            raise HTTPException(status_code=500, detail="Thiếu GEMINI_API_KEYS trong môi trường!")

        handler = GeminiHandler(
            config_path="config.yaml",
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )

        response_new = handler.generate_content(
            prompt=response_chain,
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            return_stats=True
        )

        return JSONResponse({
            "final_response": response_new.get('text', "Không có phản hồi từ mô hình."),
            "source": "\n".join(doc.get('text', {}).get('source', 'Thông tin không có sẵn') for doc in docs)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_case_data")
async def get_case_data(case_number: str):
    if not case_number:
        raise HTTPException(status_code=400, detail="Case number is required")

    target_case = next((case for case in graph_json if case["case_number"] == case_number), None)
    if not target_case:
        raise HTTPException(status_code=404, detail="Case not found")

    graph_data = {
        "nodes": [{"id": target_case["case_number"], "name": target_case["case_number"]}],
        "links": []
    }

    for related_case in target_case.get("case_relative_details", []):
        graph_data["nodes"].append({"id": related_case["case_number"], "name": related_case["case_number"]})
        graph_data["links"].append({"source": target_case["case_number"], "target": related_case["case_number"]})

    return graph_data

@app.get("/graphs")
async def graphs(case_number: str):
    if not case_number:
        raise HTTPException(status_code=400, detail="Case number is required")

    return {"case_number": case_number}

# Start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
