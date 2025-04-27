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
import pickle
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify, render_template,make_response, redirect, url_for, session

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS").split(",")

# FastAPI setup
app = FastAPI()


# Initialize embedding model
embeddings =  SentenceTransformer('hiieu/halong_embedding')  # You can change this to another model if needed


index_path = "source/faiss_index"
try:
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded with {index.ntotal} vectors.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit()

# Load metadata
metadata_path = "source/faiss_metadata.pkl"
try:
    with open(metadata_path, "rb") as f:
        metadata_dict = pickle.load(f)
    print(f"Metadata loaded with {len(metadata_dict['ids'])} documents.")
except Exception as e:
    print(f"Error loading metadata: {e}")
    exit()

def query_faiss_index(query, embeddings, index, metadata_dict, k=1):
    """Query the FAISS index and return top-k similar documents with full text."""
    try:
        # Generate query embedding
        query_embedding = embeddings.encode([query], convert_to_numpy=True)

        # Search FAISS index
        distances, indices = index.search(query_embedding, k)

        # Retrieve results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata_dict["ids"]):
                result = {
                    "id": metadata_dict["ids"][idx],
                    "metadata": metadata_dict["metadata"][idx],
                    "text": metadata_dict["texts"][idx],  # Full text, not truncated
                    "distance": distances[0][i]
                }
                results.append(result)

        return results
    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        return []

# FastAPI endpoint models
class QueryRequest(BaseModel):
    question: str

class CaseData(BaseModel):
    case_number: str


# FastAPI routes

@app.post("/query")
async def query(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Câu hỏi không hợp lệ!")

        results = query_faiss_index(question, embeddings, index, metadata_dict, 1)
          # Create context (limit to 5000 chars per document)
        context = "\n\n".join([result["text"][:5000] for result in results])
        print(context)
         # Generate response with Gemini
        response_chain = f"""
        Bạn là một chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. Nhiệm vụ của bạn là phân tích và trả lời các câu hỏi về bản án pháp luật Việt Nam theo hình thức "Bản án tương đồng và lập luận pháp lý". Bạn phải đảm bảo câu trả lời chính xác, dễ hiểu và tuân thủ các quy định pháp luật hiện hành.

        Dưới đây là câu hỏi bạn cần trả lời:
        {question}

        **Thông tin tham khảo để trả lời**:
        Nội dung bản án tìm được: {context}
        
        **Hướng dẫn trả lời**:
        1. **Cấu trúc câu trả lời**:
        - **Giới thiệu**: Đưa ra bản án tương đồng với tình huống cần giải quyết.
        - **Nội dung bản án tìm được**: Mô tả nội dung chính của bản án liên quan.
        - **Phân tích tình huống**: Phân tích tình huống cụ thể, chỉ ra các yếu tố pháp lý có liên quan.
        - **Lập luận pháp lý**: Giải thích cơ sở pháp lý, viện dẫn điều luật và công văn hướng dẫn (nếu có).
        - **Kết luận**: Xác định tội danh hoặc hướng xử lý phù hợp theo pháp luật.
        """

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
            "final_response": response_new.get('text', "Không có phản hồi từ mô hình.")        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query")
# async def query(request: QueryRequest):
#     try:
#         question = request.question.strip()
#         if not question:
#             raise HTTPException(status_code=400, detail="Câu hỏi không hợp lệ!")

#         docs = search_with_rerank(query=question, top_k=3)
#         if not docs:
#             raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu phù hợp!")

#         response_chain = prompt_template(docs, question)

#         api_keys = os.getenv('GEMINI_API_KEYS')
#         if not api_keys:
#             raise HTTPException(status_code=500, detail="Thiếu GEMINI_API_KEYS trong môi trường!")

#         handler = GeminiHandler(
#             config_path="config.yaml",
#             content_strategy=Strategy.ROUND_ROBIN,
#             key_strategy=KeyRotationStrategy.SMART_COOLDOWN
#         )

#         response_new = handler.generate_content(
#             prompt=response_chain,
#             model_name="gemini-2.0-flash-thinking-exp-01-21",
#             return_stats=True
#         )

#         return JSONResponse({
#             "final_response": response_new.get('text', "Không có phản hồi từ mô hình."),
#             "source": "\n".join(doc.get('text', {}).get('source', 'Thông tin không có sẵn') for doc in docs)
#         })
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.route("/")
def index():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

# Start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
