from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pydantic import BaseModel
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
except Exception as e:
    print(f"Error loading metadata: {e}")
    raise

def query_faiss_index(query: str,
                      embeddings: SentenceTransformer,
                      idx: faiss.Index,
                      metadata: dict,
                      k: int = 5):
    """Query the FAISS index and return top-k similar documents with full text."""
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

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Câu hỏi không hợp lệ!"}), 400

    try:
        results = query_faiss_index(question, embeddings, faiss_index, metadata_dict, k=5)
    except Exception as e:
        app.logger.exception("Error querying FAISS")
        return jsonify({"error": "Lỗi khi truy vấn FAISS: " + str(e)}), 500

    top_matching_docs = [{"id": result["id"], "source": result["metadata"]["source"], "text": result["text"][:150], "distance": result["distance"]} for result in results]

    top_matching = [{"source": result["metadata"]["source"], "text": result["text"], "distance": result["distance"]} for result in results]

    prompt = f"""
    Bạn là một chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. 
    Với kiến thức sâu rộng về các bản án tương đồng, bạn sẽ phân tích vấn đề pháp lý theo các bước chi tiết sau đây để cung cấp một câu trả lời chặt chẽ, rõ ràng và thuyết phục:

    **Câu hỏi:**  
    {question}

    **Thông tin tham khảo (Bản án tương đồng):**  
    {top_matching}

    **Hướng dẫn trả lời chi tiết:**

    1. **Giới thiệu tổng quan về các bản án tương đồng:**  
    - Trình bày tên, bối cảnh, và nguồn gốc của từng bản án, giúp người đọc nắm được sự liên quan của các vụ án trong cùng một phạm vi pháp lý.
    - Chỉ rõ loại tranh chấp và các yếu tố pháp lý chính mà các bản án này đang xét xử. Việc này giúp làm rõ sự phù hợp của các bản án với câu hỏi cần phân tích.

    2. **Nội dung chi tiết của từng bản án:**  
    - Tóm tắt sự kiện quan trọng và các vấn đề pháp lý trong từng bản án. Đặc biệt lưu ý đến những yếu tố quyết định của vụ án và lập luận pháp lý mà tòa án đã sử dụng để giải quyết tranh chấp.
    - Phân tích rõ các khía cạnh pháp lý liên quan như hợp đồng, nghĩa vụ bồi thường, trách nhiệm pháp lý của các bên trong vụ án.

    3. **Phân tích tình huống pháp lý trong từng bản án:**  
    - Phân tích các tình huống pháp lý trọng tâm của từng bản án, làm rõ các yếu tố tác động đến quyết định của tòa án.
    - So sánh các tình huống tương đồng và khác biệt giữa các bản án, từ đó đưa ra sự tương thích hoặc không tương thích với câu hỏi pháp lý.

    4. **Lập luận pháp lý trong từng bản án:**  
    - Cung cấp một phân tích chi tiết về căn cứ pháp lý mà tòa án dựa vào để đưa ra quyết định. Viện dẫn các điều luật cụ thể trong Bộ luật Dân sự, Luật Thương mại và các văn bản pháp lý khác có liên quan.
    - Trình bày cách thức áp dụng các điều luật vào tình huống thực tế của các vụ án để người đọc dễ dàng hình dung quá trình pháp lý.

    5. **Kết luận từ từng bản án:**  
    - Tóm tắt kết luận của tòa án từ mỗi bản án, nêu rõ lý do tại sao các bản án này có thể áp dụng vào tình huống câu hỏi đưa ra.
    - Chỉ ra các yếu tố cần lưu ý để áp dụng các phán quyết của tòa án vào tình huống tương tự, giúp người đọc nhận thức rõ hơn về hậu quả pháp lý và quyền lợi của các bên.

    **Lưu ý quan trọng:**  
    - Mỗi phần cần được phân tích một cách chi tiết, kết hợp giữa lý thuyết pháp lý và thực tiễn vụ án.
    - Đảm bảo các phần phân tích được liên kết chặt chẽ với nhau để tạo thành một chuỗi lập luận mạch lạc, làm rõ sự phù hợp của các bản án với câu hỏi cụ thể.
    - Trình bày rõ ràng, mạch lạc và có tính thuyết phục cao, từ đó giúp người đọc dễ dàng áp dụng các kiến thức này vào tình huống pháp lý tương tự.
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

    return jsonify({"final_response": answer, "top_matching_documents": top_matching_docs})

@app.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

if __name__ == "__main__":
    # debug=True will auto‑reload on code changes
    app.run(host="0.0.0.0", port=5000, debug=True)
