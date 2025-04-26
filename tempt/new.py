from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss
import numpy as np
import pickle
import os
from datetime import datetime
from dotenv import load_dotenv
from gemini_handler import GeminiHandler, GenerationConfig, Strategy, KeyRotationStrategy
from sentence_transformers import SentenceTransformer

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
    # Kiểm tra trường type
    for meta in metadata_dict["metadata"]:
        if "type" not in meta:
            meta["type"] = "banan"
except Exception as e:
    print(f"Error loading metadata: {e}")
    raise

# Load summarized metadata
metadata_path_sum = "source/summarized_faiss_metadata.pkl"
try:
    with open(metadata_path_sum, "rb") as f:
        metadata_path_sum_dict = pickle.load(f)
    print(f"Summarized metadata loaded with {len(metadata_path_sum_dict['ids'])} documents.")
    for meta in metadata_path_sum_dict["metadata"]:
        if "type" not in meta:
            meta["type"] = "banan"
except Exception as e:
    print(f"Error loading summarized metadata: {e}")
    raise

# Load FAISS index cho án lệ
anle_index_path = "source/faiss_index_anle.index"
try:
    faiss_anle_index = faiss.read_index(anle_index_path)
    print(f"FAISS án lệ index loaded with {faiss_anle_index.ntotal} vectors.")
except Exception as e:
    print(f"Error loading FAISS án lệ index: {e}")
    raise

# Load metadata cho án lệ
anle_metadata_path = "source/metadata_anle.pkl"
try:
    with open(anle_metadata_path, "rb") as f:
        anle_metadata_dict = pickle.load(f)
    print(f"Án lệ metadata loaded with {len(anle_metadata_dict['ids'])} documents.")
    for meta in anle_metadata_dict["metadata"]:
        if "type" not in meta:
            meta["type"] = "anle"
except Exception as e:
    print(f"Error loading án lệ metadata: {e}")
    raise

def query_faiss_index(query: str, embeddings: SentenceTransformer, idx: faiss.Index, metadata: dict, k: int = 5, doc_type: str = None, max_threshold: float = 0.8):
    """Query FAISS index and return top-k similar documents, filtered by type and dynamic distance threshold."""
    query_emb = embeddings.encode([query], convert_to_numpy=True)
    distances, indices = idx.search(query_emb, k)
    results = []

    # Tính ngưỡng động dựa trên phân vị 75
    if len(distances[0]) > 0:
        distance_threshold = min(max_threshold, np.percentile(distances[0], 50))
    else:
        distance_threshold = max_threshold

    for dist, i in zip(distances[0], indices[0]):
        if 0 <= i < len(metadata["ids"]):  # Bỏ điều kiện dist <= distance_threshold để kiểm tra
            result = {
                "id": metadata["ids"][i],
                "metadata": metadata["metadata"][i],
                "text": metadata["texts"][i][:200],  # Giới hạn để log dễ đọc
                "distance": float(dist)
            }
            if doc_type is None or result["metadata"].get("type") == doc_type:
                results.append(result)
    
    print(f"Query FAISS ({doc_type}): {len(results)} results found for query '{query}' with threshold {distance_threshold:.4f}")
    print(f"Distances: {list(distances[0])}")
    return results[:k]

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Câu hỏi không hợp lệ!"}), 400

    try:
        # Truy vấn bản án, án lệ
        banan_results = query_faiss_index(
            question, embeddings, faiss_index, metadata_dict, k=5, doc_type="banan", max_threshold=0.8
        )
        banan_sum_results = query_faiss_index(
            question, embeddings, faiss_index, metadata_path_sum_dict, k=5, doc_type="banan", max_threshold=0.8
        )
        # Truy vấn án lệ
        anle_results = query_faiss_index(
            question, embeddings, faiss_anle_index, anle_metadata_dict, k=5, doc_type="anle", max_threshold=0.8
        )

    except Exception as e:
        app.logger.exception("Error querying FAISS")
        return jsonify({"error": "Lỗi khi truy vấn FAISS: " + str(e)}), 500

    # Chuẩn bị dữ liệu bản án, án lệ
    top_banan_docs = [
        {"source": result["metadata"]["source"], "text": result["text"], "distance": result["distance"]}
        for result in banan_results
    ]
    top_banan_sum = [
        {"source": result["metadata"]["source"], "text": result["text"], "distance": result["distance"]}
        for result in banan_sum_results
    ]
    combined_banan_docs = top_banan_sum + top_banan_docs

    # Chuẩn bị dữ liệu án lệ
    top_anle_docs = [
        {"source": result["metadata"]["source"], "text": result["text"], "distance": result["distance"]}
        for result in anle_results
    ]

    # Logging dữ liệu
    print(f"Top bản án, án lệ: {top_banan_docs}")
    print(f"Top án lệ: {top_anle_docs}")

    # Prompt
    prompt = f"""
Bạn là chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. Bạn sẽ phân tích vấn đề pháp lý theo các bước chi tiết dưới đây để cung cấp câu trả lời chặt chẽ, rõ ràng, và thuyết phục, đảm bảo đề cập đến cả bản án, án lệ và án lệ.

**Câu hỏi:**  
{question}

**Sử dụng hông tin tham khảo (bản án, án lệ tương đồng) là bắt buộc:**  
{combined_banan_docs if combined_banan_docs else "Không tìm thấy bản án, án lệ phù hợp. Vui lòng phân tích dựa trên các quy định pháp luật hiện hành."}

**Sử dụng thông tin tham khảo (Án lệ tương đồng) là bắt buộc:**  
{top_anle_docs if top_anle_docs else "Không tìm thấy án lệ phù hợp. Vui lòng phân tích dựa trên các quy định pháp luật hiện hành."}

**Hướng dẫn trả lời chi tiết:**

1. **Tổng quan về các bản án, án lệ tương đồng:**  
   - Trình bày ngắn gọn tên, bối cảnh, và nguồn gốc của từng bản án, án lệ, làm rõ sự liên quan đến vấn đề pháp lý được đặt ra.  
   - Xác định loại tranh chấp (hợp đồng, dân sự, thương mại, v.v.) và các yếu tố pháp lý trọng tâm được xét xử, nhấn mạnh tính phù hợp với câu hỏi.

2. **Nội dung chi tiết của từng bản án, án lệ:**  
   - Tóm lược các sự kiện chính, vấn đề pháp lý, và lập luận của tòa án trong từng bản án, án lệ.  
   - Phân tích các yếu tố quyết định phán quyết, bao gồm hợp đồng, nghĩa vụ bồi thường, hoặc trách nhiệm pháp lý của các bên, với trọng tâm vào các khía cạnh pháp lý cốt lõi.

3. **Phân tích tình huống pháp lý:**  
   - Làm rõ các tình huống pháp lý trọng tâm trong từng bản án, án lệ, nêu bật yếu tố ảnh hưởng đến quyết định của tòa án.  
   - So sánh điểm tương đồng và khác biệt giữa các bản án, án lệ, đánh giá mức độ áp dụng vào câu hỏi pháp lý đang phân tích.

4. **Lập luận pháp lý:**  
   - Phân tích chi tiết căn cứ pháp lý mà tòa án sử dụng, viện dẫn cụ thể các điều luật từ Bộ luật Dân sự, Luật Thương mại, hoặc các văn bản pháp luật liên quan.  
   - Giải thích cách áp dụng các điều luật vào tình huống thực tế của bản án, án lệ, đảm bảo dễ hiểu và minh họa rõ ràng quá trình lập luận.

5. **Kết luận từ các bản án, án lệ:**  
   - Tóm tắt phán quyết của từng bản án, án lệ, làm rõ lý do chúng có thể áp dụng vào tình huống của câu hỏi.  
   - Chỉ ra các yếu tố cần lưu ý khi áp dụng phán quyết vào tình huống tương tự, nhấn mạnh quyền lợi, nghĩa vụ, và hậu quả pháp lý của các bên.

**Lưu ý quan trọng:**  
- Không dùng từ giả sử, ví dụ.
- Bắt buộc có đủ thông tin bản án, án lệ.
- Phân tích phải kết hợp chặt chẽ giữa lý thuyết pháp lý và thực tiễn vụ án, đảm bảo tính chi tiết và thực tiễn.  
- Các phần cần liên kết mạch lạc, tạo thành chuỗi lập luận thuyết phục, làm nổi bật sự phù hợp của các bản án, án lệ với câu hỏi.  
- Trình bày rõ ràng, súc tích, sử dụng ngôn ngữ pháp lý chính xác, giúp người đọc dễ dàng áp dụng vào tình huống pháp lý thực tế.
"""

    try:
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
    except Exception as e:
        app.logger.exception("Error generating with Gemini")
        return jsonify({"error": "Lỗi khi gọi Gemini: " + str(e)}), 500

    return jsonify({
        "final_response": answer,
        "top_banan_documents": top_banan_sum,
        "top_anle_documents": top_anle_docs
    })

@app.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)