from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from pydantic import BaseModel
import faiss
import numpy as np
import pickle
import os
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load environment
load_dotenv()
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "")
if not GEMINI_API_KEYS:
    raise RuntimeError("Thiếu GEMINI_API_KEYS trong môi trường!")

# Flask setup
app = Flask(__name__)
CORS(app)

# Embedding model
embeddings = SentenceTransformer('hiieu/halong_embedding')

# Load FAISS index
index_path = "source/faiss_index"
try:
    faiss_index = faiss.read_index(index_path)
    print(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    raise

# Load metadata
metadata_path = "source/faiss_metadata.pkl"
try:
    with open(metadata_path, "rb") as f:
        metadata_dict = pickle.load(f)
    print(f"Metadata loaded with {len(metadata_dict['ids'])} documents.")
except Exception as e:
    print(f"Error loading metadata: {e}")
    raise

# Load summarized metadata
metadata_path_sum = "source/summarized_faiss_metadata.pkl"
try:
    with open(metadata_path_sum, "rb") as f:
        metadata_path_sum_dict = pickle.load(f)
    print(f"Summarized metadata loaded with {len(metadata_path_sum_dict['ids'])} documents.")
except Exception as e:
    print(f"Error loading summarized metadata: {e}")
    raise

# Initialize BM25
def prepare_bm25_corpus(metadata):
    corpus = [str(doc) for doc in metadata["texts"]]
    tokenized_corpus = [doc.split() for doc in corpus]
    return tokenized_corpus, corpus

tokenized_corpus, corpus = prepare_bm25_corpus(metadata_dict)
bm25 = BM25Okapi(tokenized_corpus)

# Load Cross-Encoder for Re-Ranking
rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)

# FAISS query function
def query_faiss_index(query: str, embeddings: SentenceTransformer, idx: faiss.Index, metadata: dict, k: int = 10):
    query_emb = embeddings.encode([query], convert_to_numpy=True)
    distances, indices = idx.search(query_emb, k)
    results = []
    for dist, i in zip(distances[0], indices[0]):
        if 0 <= i < len(metadata["ids"]):
            results.append({
                "id": metadata["ids"][i],
                "metadata": metadata["metadata"][i],
                "text": metadata["texts"][i],
                "distance": float(dist)
            })
    return results

# BM25 search
def bm25_search(query: str, top_k: int = 10):
    try:
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            {
                "id": metadata_dict["ids"][i],
                "metadata": metadata_dict["metadata"][i],
                "text": metadata_dict["texts"][i],
                "bm25_score": float(scores[i])
            }
            for i in top_indices if scores[i] > 0
        ]
        return results
    except Exception as e:
        print(f"Error during BM25 search: {e}")
        return []

# TF-IDF Re-Ranking
def tfidf_rerank(query: str, search_results: list, top_k: int = 10):
    documents = [str(result["text"]) for result in search_results]
    documents.append(query)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    for i, result in enumerate(search_results):
        result["tfidf_score"] = float(cosine_similarities[i])
    return sorted(search_results, key=lambda x: x["tfidf_score"], reverse=True)[:top_k]

# Cross-Encoder Re-Ranking
def rerank(query: str, search_results: list, batch_size: int = 8):
    scores = []
    for i in range(0, len(search_results), batch_size):
        batch_results = search_results[i:i + batch_size]
        batch_inputs = [
            tokenizer.encode_plus(
                query,
                str(result["text"]),
                return_tensors="pt",
                truncation=True,
                max_length=512
            ) for result in batch_results
        ]
        input_ids = torch.cat([inputs["input_ids"] for inputs in batch_inputs])
        attention_mask = torch.cat([inputs["attention_mask"] for inputs in batch_inputs])
        with torch.no_grad():
            batch_scores = rerank_model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            scores.extend(batch_scores.tolist())
    for i, result in enumerate(search_results):
        result["rerank_score"] = float(scores[i])
    return sorted(search_results, key=lambda x: x["rerank_score"], reverse=True)

# Combined search with FAISS, BM25, TF-IDF, and Cross-Encoder
def search_with_rerank(query: str, top_k: int = 3):
    # Step 1: FAISS search
    faiss_results = query_faiss_index(query, embeddings, faiss_index, metadata_dict, k=top_k * 2)
    if not faiss_results:
        return []

    # Step 2: BM25 search
    bm25_results = bm25_search(query, top_k=top_k * 2)
    if not bm25_results:
        bm25_results = []

    # Step 3: Combine results
    combined_results = faiss_results + bm25_results
    unique_results = []
    seen_texts = set()
    for result in combined_results:
        text = str(result["text"])
        if text not in seen_texts:
            unique_results.append(result)
            seen_texts.add(text)

    # Step 4: TF-IDF Re-Ranking
    tfidf_results = tfidf_rerank(query, unique_results, top_k=top_k * 2)

    # Step 5: Cross-Encoder Re-Ranking
    reranked_results = rerank(query, tfidf_results)[:top_k]

    return reranked_results

# Prompt template
def prompt_template(docs: list, question: str) -> str:
    top_matching_docs = [
        {
            "id": doc["id"],
            "source": doc["metadata"]["source"],
            "text": doc["text"],
            "distance": doc.get("distance", 0.0)  # Default to 0.0 if distance is missing
        } for doc in docs
    ]
    doc_strings = [
        f"ID: {doc['id']}, Source: {doc['source']}, Text: {doc['text'][:500]}..., Distance: {doc['distance']}"
        for doc in top_matching_docs
    ]
    docs_context = "\n".join(doc_strings)

    prompt = f"""
Bạn là chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam, sở hữu kiến thức sâu rộng về các bản án tương đồng. Bạn sẽ phân tích vấn đề pháp lý theo các bước chi tiết dưới đây để cung cấp câu trả lời chặt chẽ, rõ ràng, và thuyết phục.

**Câu hỏi:**  
{question}

**Thông tin tham khảo (Bản án tương đồng):**  
{docs_context}

**Hướng dẫn trả lời chi tiết:**

1. **Tổng quan về các bản án tương đồng:**  
   - Trình bày ngắn gọn tên, bối cảnh, và nguồn gốc của từng bản án, làm rõ sự liên quan đến vấn đề pháp lý được đặt ra.  
   - Xác định loại tranh chấp (hợp đồng, dân sự, thương mại, v.v.) và các yếu tố pháp lý trọng tâm được xét xử, nhấn mạnh tính phù hợp với câu hỏi.

2. **Nội dung chi tiết của từng bản án:**  
   - Tóm lược các sự kiện chính, vấn đề pháp lý, và lập luận của tòa án trong từng bản án.  
   - Phân tích các yếu tố quyết định phán quyết, bao gồm hợp đồng, nghĩa vụ bồi thường, hoặc trách nhiệm pháp lý của các bên, với trọng tâm vào các khía cạnh pháp lý cốt lõi.

3. **Phân tích tình huống pháp lý:**  
   - Làm rõ các tình huống pháp lý trọng tâm trong từng bản án, nêu bật yếu tố ảnh hưởng đến quyết định của tòa án.  
   - So sánh điểm tương đồng và khác biệt giữa các bản án, đánh giá mức độ áp dụng vào câu hỏi pháp lý đang phân tích.

4. **Lập luận pháp lý:**  
   - Phân tích chi tiết căn cứ pháp lý mà tòa án sử dụng, viện dẫn cụ thể các điều luật từ Bộ luật Dân sự, Luật Thương mại, hoặc các văn bản pháp luật liên quan.  
   - Giải thích cách áp dụng các điều luật vào tình huống thực tế của bản án, đảm bảo dễ hiểu và minh họa rõ ràng quá trình lập luận.

5. **Kết luận từ các bản án:**  
   - Tóm tắt phán quyết của từng bản án, làm rõ lý do chúng có thể áp dụng vào tình huống của câu hỏi.  
   - Chỉ ra các yếu tố cần lưu ý khi áp dụng phán quyết vào tình huống tương tự, nhấn mạnh quyền lợi, nghĩa vụ, và hậu quả pháp lý của các bên.

**Lưu ý quan trọng:**  
- Phân tích phải kết hợp chặt chẽ giữa lý thuyết pháp lý và thực tiễn vụ án, đảm bảo tính chi tiết và thực tiễn.  
- Các phần cần liên kết mạch lạc, tạo thành chuỗi lập luận thuyết phục, làm nổi bật sự phù hợp của các bản án với câu hỏi.  
- Trình bày rõ ràng, súc tích, sử dụng ngôn ngữ pháp lý chính xác, giúp người đọc dễ dàng áp dụng vào tình huống pháp lý thực tế.
"""
    return prompt

# Query endpoint
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Câu hỏi không hợp lệ!"}), 400

    try:
        # Search with reranking
        results = search_with_rerank(question, top_k=5)
        if not results:
            return jsonify({"error": "Không tìm thấy tài liệu phù hợp!"}), 404

        top_matching = [
            {"source": result["metadata"]["source"], "text": result["text"][:100], "distance": result.get("distance", 0.0)}
            for result in results
        ]

        # Generate prompt
        prompt = prompt_template(results, question)

        # Call Gemini
        from gemini_handler import GeminiHandler, GenerationConfig, Strategy, KeyRotationStrategy
        handler = GeminiHandler(
            config_path="config.yaml",
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )
        gen = handler.generate_content(
            prompt=prompt,
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            return_stats=False
        )
        answer = gen.get("text", "Không có phản hồi từ mô hình.")

        return jsonify({
            "final_response": answer,
            "top_matching_documents": top_matching
        })

    except Exception as e:
        app.logger.exception("Error processing query")
        return jsonify({"error": f"Lỗi khi xử lý truy vấn: {str(e)}"}), 500

# Home endpoint
@app.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)