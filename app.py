
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
import logging
from rank_bm25 import BM25Okapi  # Import BM25
from nltk.tokenize import word_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
except Exception as e:
    logger.error(f"Error loading FAISS index: {e}")
    raise

# Load metadata
metadata_path = "source/faiss_metadata.pkl"
try:
    with open(metadata_path, "rb") as f:
        metadata_dict = pickle.load(f)
    logger.info(f"Metadata loaded with {len(metadata_dict['ids'])} documents.")
    for i, meta in enumerate(metadata_dict["metadata"]):
        if "type" not in meta:
            meta["type"] = "banan"
        meta["case_summary"] = meta.get("case_summary", "No summary available")
        meta["legal_issues"] = meta.get("legal_issues", "No legal issues specified")
        meta["court_reasoning"] = meta.get("court_reasoning", "No reasoning provided")
        meta["decision"] = meta.get("decision", "No decision available")
        meta["relevant_laws"] = meta.get("relevant_laws", "No laws cited")
        if not metadata_dict["texts"][i]:
            logger.warning(f"Empty text field for document ID {metadata_dict['ids'][i]}")
except Exception as e:
    logger.error(f"Error loading metadata: {e}")
    raise

# Load summarized metadata
metadata_path_sum = "source/summarized_faiss_metadata.pkl"
try:
    with open(metadata_path_sum, "rb") as f:
        metadata_path_sum_dict = pickle.load(f)
    logger.info(f"Summarized metadata loaded with {len(metadata_path_sum_dict['ids'])} documents.")
    for i, meta in enumerate(metadata_path_sum_dict["metadata"]):
        if "type" not in meta:
            meta["type"] = "banan"
        meta["case_summary"] = meta.get("case_summary", "No summary available")
        meta["legal_issues"] = meta.get("legal_issues", "No legal issues specified")
        meta["court_reasoning"] = meta.get("court_reasoning", "No reasoning provided")
        meta["decision"] = meta.get("decision", "No decision available")
        meta["relevant_laws"] = meta.get("relevant_laws", "No laws cited")
        if not metadata_path_sum_dict["texts"][i]:
            logger.warning(f"Empty text field for summarized document ID {metadata_path_sum_dict['ids'][i]}")
except Exception as e:
    logger.error(f"Error loading summarized metadata: {e}")
    raise

# Load FAISS index for precedents (án lệ)
anle_index_path = "source/faiss_index_anle.index"
try:
    faiss_anle_index = faiss.read_index(anle_index_path)
    logger.info(f"FAISS án lệ index loaded with {faiss_anle_index.ntotal} vectors.")
except Exception as e:
    logger.error(f"Error loading FAISS án lệ index: {e}")
    raise

# Load metadata for precedents
anle_metadata_path = "source/metadata_anle.pkl"
try:
    with open(anle_metadata_path, "rb") as f:
        anle_metadata_dict = pickle.load(f)
    logger.info(f"Án lệ metadata loaded with {len(anle_metadata_dict['ids'])} documents.")
    for i, meta in enumerate(anle_metadata_dict["metadata"]):
        if "type" not in meta:
            meta["type"] = "anle"
        meta["case_summary"] = meta.get("case_summary", "No summary available")
        meta["legal_issues"] = meta.get("legal_issues", "No legal issues specified")
        meta["court_reasoning"] = meta.get("court_reasoning", "No reasoning provided")
        meta["decision"] = meta.get("decision", "No decision available")
        meta["relevant_laws"] = meta.get("relevant_laws", "No laws cited")
        if not anle_metadata_dict["texts"][i]:
            logger.warning(f"Empty text field for án lệ document ID {anle_metadata_dict['ids'][i]}")
except Exception as e:
    logger.error(f"Error loading án lệ metadata: {e}")
    raise

# Initialize BM25 indices
def tokenize_text(text):
    """Tokenize text using NLTK."""
    return word_tokenize(text.lower())

# Create BM25 indices for each dataset
try:
    tokenized_banan_texts = [tokenize_text(text) for text in metadata_dict["texts"]]
    bm25_banan = BM25Okapi(tokenized_banan_texts)
    logger.info("BM25 index initialized for bản án.")

    tokenized_banan_sum_texts = [tokenize_text(text) for text in metadata_path_sum_dict["texts"]]
    bm25_banan_sum = BM25Okapi(tokenized_banan_sum_texts)
    logger.info("BM25 index initialized for bản án tóm tắt.")

    tokenized_anle_texts = [tokenize_text(text) for text in anle_metadata_dict["texts"]]
    bm25_anle = BM25Okapi(tokenized_anle_texts)
    logger.info("BM25 index initialized for án lệ.")
except Exception as e:
    logger.error(f"Error initializing BM25 indices: {e}")
    raise

def query_bm25_index(query: str, bm25: BM25Okapi, metadata: dict, k: int = 5, doc_type: str = None, score_threshold: float = 0.0):
    """Query BM25 index and return top-k similar documents."""
    tokenized_query = tokenize_text(query)
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    results = []

    for idx in top_k_indices:
        if scores[idx] >= score_threshold and 0 <= idx < len(metadata["ids"]):
            result = {
                "id": metadata["ids"][idx],
                "metadata": metadata["metadata"][idx],
                "text": metadata["texts"][idx],
                "score": float(scores[idx]),
                "case_summary": metadata["metadata"][idx].get("case_summary", "No summary"),
                "legal_issues": metadata["metadata"][idx].get("legal_issues", "No issues"),
                "court_reasoning": metadata["metadata"][idx].get("court_reasoning", "No reasoning"),
                "decision": metadata["metadata"][idx].get("decision", "No decision"),
                "relevant_laws": metadata["metadata"][idx].get("relevant_laws", "No laws")
            }
            if doc_type is None or result["metadata"].get("type") == doc_type:
                results.append(result)

    logger.info(f"Query BM25 ({doc_type}): {len(results)} results found for query '{query}' with threshold {score_threshold:.4f}")
    logger.debug(f"BM25 scores: {[float(scores[i]) for i in top_k_indices]}")
    return results[:k]

def query_hybrid_index(query: str, embeddings: SentenceTransformer, faiss_idx: faiss.Index, bm25: BM25Okapi, metadata: dict, k: int = 5, doc_type: str = None, faiss_weight: float = 0.5, bm25_weight: float = 0.5):
    """Combine FAISS and BM25 results using weighted scoring."""
    # FAISS search
    faiss_results = query_faiss_index(query, embeddings, faiss_idx, metadata, k=k, doc_type=doc_type, max_threshold=0.8)
    # BM25 search
    bm25_results = query_bm25_index(query, bm25, metadata, k=k, doc_type=doc_type, score_threshold=0.8)

    # Normalize scores
    faiss_scores = {res["id"]: 1.0 - res["distance"] for res in faiss_results}  # Convert distance to similarity
    bm25_scores = {res["id"]: res["score"] for res in bm25_results}
    
    # Combine results
    combined_results = {}
    all_ids = set(faiss_scores.keys()).union(bm25_scores.keys())
    
    max_bm25_score = max(bm25_scores.values(), default=1.0) or 1.0
    for id in all_ids:
        faiss_score = faiss_scores.get(id, 0.0)
        bm25_score = bm25_scores.get(id, 0.0) / max_bm25_score  # Normalize BM25 score
        combined_score = faiss_weight * faiss_score + bm25_weight * bm25_score
        combined_results[id] = combined_score

    # Sort by combined score
    sorted_ids = sorted(combined_results, key=combined_results.get, reverse=True)[:k]
    results = []
    for id in sorted_ids:
        # Find the result from FAISS or BM25
        for res in faiss_results + bm25_results:
            if res["id"] == id:
                results.append(res)
                break
    
    logger.info(f"Hybrid search ({doc_type}): {len(results)} results after combining FAISS and BM25")
    return results

def query_faiss_index(query: str, embeddings: SentenceTransformer, idx: faiss.Index, metadata: dict, k: int = 5, doc_type: str = None, max_threshold: float = 0.8):
    """Query FAISS index and return top-k similar documents (unchanged)."""
    query_emb = embeddings.encode([query], convert_to_numpy=True)
    distances, indices = idx.search(query_emb, k)
    results = []

    if len(distances[0]) > 0:
        distance_threshold = min(max_threshold, np.percentile(distances[0], 50))
    else:
        distance_threshold = max_threshold

    for dist, i in zip(distances[0], indices[0]):
        if 0 <= i < len(metadata["ids"]):
            result = {
                "id": metadata["ids"][i],
                "metadata": metadata["metadata"][i],
                "text": metadata["texts"][i],
                "distance": float(dist),
                "case_summary": metadata["metadata"][i].get("case_summary", "No summary"),
                "legal_issues": metadata["metadata"][i].get("legal_issues", "No issues"),
                "court_reasoning": metadata["metadata"][i].get("court_reasoning", "No reasoning"),
                "decision": metadata["metadata"][i].get("decision", "No decision"),
                "relevant_laws": metadata["metadata"][i].get("relevant_laws", "No laws")
            }
            if doc_type is None or result["metadata"].get("type") == doc_type:
                results.append(result)
    
    logger.info(f"Query FAISS ({doc_type}): {len(results)} results found for query '{query}' with threshold {distance_threshold:.4f}")
    logger.debug(f"Distances: {list(distances[0])}")
    return results[:k]

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Câu hỏi không hợp lệ!"}), 400

    try:
        # Query judgments and precedents using hybrid search
        banan_results = query_hybrid_index(
            question, embeddings, faiss_index, bm25_banan, metadata_dict, k=5, doc_type="banan", faiss_weight=0.5, bm25_weight=0.5
        )
        banan_sum_results = query_hybrid_index(
            question, embeddings, faiss_index, bm25_banan_sum, metadata_path_sum_dict, k=5, doc_type="banan", faiss_weight=0.5, bm25_weight=0.5
        )
        anle_results = query_hybrid_index(
            question, embeddings, faiss_anle_index, bm25_anle, anle_metadata_dict, k=5, doc_type="anle", faiss_weight=0.5, bm25_weight=0.5
        )

    except Exception as e:
        logger.exception("Error querying hybrid index")
        return jsonify({"error": "Lỗi khi truy vấn hybrid index: " + str(e)}), 500

    # Prepare judgment and precedent data
    top_banan_docs = [
        {
            "source": result["metadata"]["source"],
            "text": result["text"],
            "distance": result.get("distance", 0.0),
            "case_summary": result["case_summary"],
            "legal_issues": result["legal_issues"],
            "court_reasoning": result["court_reasoning"],
            "decision": result["decision"],
            "relevant_laws": result["relevant_laws"]
        }
        for result in banan_results
    ]
    top_banan_sum = [
        {
            "source": result["metadata"]["source"],
            "text": result["text"],
            "distance": result.get("distance", 0.0),
            "case_summary": result["case_summary"],
            "legal_issues": result["legal_issues"],
            "court_reasoning": result["court_reasoning"],
            "decision": result["decision"],
            "relevant_laws": result["relevant_laws"]
        }
        for result in banan_sum_results
    ]
    top_anle_docs = [
        {
            "source": result["metadata"]["source"],
            "text": result["text"],
            "distance": result.get("distance", 0.0),
            "case_summary": result["case_summary"],
            "legal_issues": result["legal_issues"],
            "court_reasoning": result["court_reasoning"],
            "decision": result["decision"],
            "relevant_laws": result["relevant_laws"]
        }
        for result in anle_results
    ]

    # Log retrieved data
    logger.debug(f"Top bản án: {top_banan_docs}")
    logger.debug(f"Top bản án tóm tắt: {top_banan_sum}")
    logger.debug(f"Top án lệ: {top_anle_docs}")

    # Prompt (unchanged)
    prompt = f"""
Bạn là chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. Bạn sẽ phân tích vấn đề pháp lý theo các bước chi tiết dưới đây để cung cấp câu trả lời chặt chẽ, rõ ràng, và thuyết phục, đảm bảo đề cập đến cả bản án và án lệ khi có sẵn.

**Câu hỏi:**  
{question}

**Thông tin tham khảo (bản án tương đồng):**  
{top_banan_sum if top_banan_sum else "Không tìm thấy bản án phù hợp. Phân tích dựa trên các quy định pháp luật hiện hành và nguyên tắc pháp lý chung."}

**Thông tin tham khảo (án lệ tương đồng):**  
{top_anle_docs if top_anle_docs else "Không tìm thấy án lệ phù hợp. Phân tích dựa trên các quy định pháp luật hiện hành và nguyên tắc pháp lý chung."}

**Hướng dẫn trả lời chi tiết:**

1. **Tổng quan về các bản án, án lệ tương đồng:**  
   - Chỉ ra rõ thông tin tham khảo từ bản án, án lệ tương đồng đã được cung cấp.
   - Nếu có thông tin chi tiết về bản án hoặc án lệ, trình bày ngắn gọn tên, bối cảnh, và nguồn gốc, làm rõ sự liên quan đến vấn đề pháp lý được đặt ra.  
   - Xác định loại tranh chấp (hợp đồng, dân sự, thương mại, v.v.) và các yếu tố pháp lý trọng tâm, nhấn mạnh tính phù hợp với câu hỏi.  
   - Nếu thông tin bản án hoặc án lệ chỉ có tên hoặc số hiệu, nêu rõ rằng thông tin chi tiết không khả dụng và chuyển sang phân tích dựa trên quy định pháp luật hiện hành.

2. **Nội dung chi tiết của từng bản án, án lệ:**  
   - Nếu có thông tin chi tiết, tóm lược các sự kiện chính, vấn đề pháp lý, và lập luận của tòa án trong từng bản án, án lệ.  
   - Phân tích các yếu tố quyết định phán quyết, bao gồm hợp đồng, nghĩa vụ bồi thường, hoặc trách nhiệm pháp lý của các bên.  
   - Nếu thiếu chi tiết, nêu rõ hạn chế và thay bằng phân tích các nguyên tắc pháp lý liên quan đến câu hỏi.

3. **Phân tích tình huống pháp lý:**  
   - Nếu có bản án hoặc án lệ, làm rõ các tình huống pháp lý trọng tâm, nêu bật yếu tố ảnh hưởng đến quyết định của tòa án.  
   - So sánh điểm tương đồng và khác biệt giữa các bản án, án lệ, đánh giá mức độ áp dụng vào câu hỏi pháp lý.  
   - Nếu không có thông tin chi tiết, phân tích tình huống dựa trên các quy định pháp luật hiện hành (ví dụ: Bộ luật Dân sự, Luật Thương mại).

4. **Lập luận pháp lý:**  
   - Phân tích chi tiết căn cứ pháp lý, viện dẫn cụ thể các điều luật từ Bộ luật Dân sự, Luật Thương mại, hoặc các văn bản pháp luật liên quan.  
   - Giải thích cách áp dụng các điều luật vào tình huống thực tế, đảm bảo dễ hiểu và minh họa rõ ràng quá trình lập luận.  
   - Nếu có bản án hoặc án lệ, liên hệ với lập luận của tòa án; nếu không, xây dựng lập luận dựa trên luật và nguyên tắc pháp lý.

5. **Kết luận từ các bản án, án lệ:**  
   - Nếu có thông tin chi tiết, tóm tắt phán quyết của từng bản án, án lệ, làm rõ lý do chúng có thể áp dụng vào tình huống của câu hỏi.  
   - Nếu thiếu chi tiết, đưa ra kết luận dựa trên phân tích pháp lý, nhấn mạnh quyền lợi, nghĩa vụ, và hậu quả pháp lý của các bên.  
   - Chỉ ra các yếu tố cần lưu ý khi áp dụng vào tình huống tương tự.

**Lưu ý quan trọng:**  
- Không dùng từ giả sử, ví dụ. 
- Bỏ phần chào hỏi, giới thiệu mình là ai. 
- Không cần nêu quy trình phân tích, không giới thiệu 30 năm kinh nghiệm.
- Nếu xác định được bản án hay án lệ không phù hợp hãy bỏ qua, không đề cập đến trong câu trả lời.
- Phân tích phải kết hợp chặt chẽ giữa lý thuyết pháp lý và thực tiễn vụ án (nếu có), đảm bảo tính chi tiết và thực tiễn.  
- Nếu thông tin bản án hoặc án lệ không đủ chi tiết, tập trung vào phân tích pháp lý dựa trên các quy định pháp luật hiện hành.  
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
        logger.exception("Error generating with Gemini")
        return jsonify({"error": "Lỗi khi gọi Gemini: " + str(e)}), 500

    
    return jsonify({
        "final_response": answer,
        "top_banan_documents": top_banan_sum,
        "top_anle_documents": top_anle_docs
    })


@app.route("/draft_judgment", methods=["POST"])
def draft_judgment():
    data = request.get_json(silent=True) or {}
    case_details = data.get("case_details", "").strip()
    if not case_details:
        return jsonify({"error": "Thông tin vụ án không hợp lệ!"}), 400

    try:
        # Query relevant judgments and precedents
        banan_results = query_faiss_index(
            case_details, embeddings, faiss_index, metadata_dict, k=2, doc_type="banan", max_threshold=0.8
        )
        anle_results = query_faiss_index(
            case_details, embeddings, faiss_anle_index, anle_metadata_dict, k=2, doc_type="anle", max_threshold=0.8
        )

        # Prepare data for prompt
        top_banan_docs = [
            {
                "source": result["metadata"]["source"],
                "case_summary": result["case_summary"],
                "legal_issues": result["legal_issues"],
                "court_reasoning": result["court_reasoning"],
                "decision": result["decision"],
                "relevant_laws": result["relevant_laws"]
            }
            for result in banan_results
        ]
        top_anle_docs = [
            {
                "source": result["metadata"]["source"],
                "case_summary": result["case_summary"],
                "legal_issues": result["legal_issues"],
                "court_reasoning": result["court_reasoning"],
                "decision": result["decision"],
                "relevant_laws": result["relevant_laws"]
            }
            for result in anle_results
        ]

        # Prompt for drafting judgment
        prompt = f"""
Bạn là trợ lý pháp lý thông minh, có nhiệm vụ **hỗ trợ soạn thảo bản án hành chính sơ thẩm** theo đúng **Mẫu số 22-HC**, ban hành kèm theo **Nghị quyết số 02/2017/NQ-HĐTP ngày 13/01/2017** của Hội đồng Thẩm phán TAND Tối cao.

Hãy giúp tôi **soạn bản án đầy đủ**, trình bày rõ ràng, theo đúng định dạng mẫu, với các phần cụ thể như sau:

Dưới đây là **thông tin vụ án** để bạn dựa vào đó và soạn bản án:

{case_details}

- Tên Tòa án.... (1) nằm trên header bên trái và in đậm, in hoa toàn bộ.
- Số bản án và năm ban hành (2) nằm trên header bên trái  và in đậm.
- Ngày tuyên án (3) nằm trên header bên trái và in đậm.
- V/v... (4) nằm trên header bên trái và in đậm.
- Quốc hiệu "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" in đậm, nằm trên header bên phải ngang hàng với (1),(2),(3),(4) và căn giữa.
- Tiêu ngữ "Độc lập - Tự do - Hạnh phúc" được in đậm, không in hoa toàn bộ, chỉ in hoa chữ cái đầu mỗi từ,  nằm trên header về phía bên phải ngang hàng với (1)(2)(3)(4) và căn giữa.
- Bắt buộc có dòng chữ "NHÂN DANH" chứ không phải "NHÂN DÂN" nằm giữa, in đậm, viết in hoa toàn bộ.
- Bắt buộc có dòng chữ "NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" nằm giữa, in đậm, viết in hoa toàn bộ.
- Bắt buộc có dòng chữ "TÒA ÁN NHÂN DÂN ...." nằm giữa, in đậm, viết in hoa toàn bộ.

### -Thành phần Hội đồng xét xử sơ thẩm gồm có: ** phần này được in đậm**(6)
Thẩm phán - Chủ toạ phiên tòa: Ông (Bà)... 
Thẩm phán: Ông (Bà)...
Các Hội thẩm nhân dân:
1. Ông (Bà)... 
2. Ông (Bà)... 
3. Ông (Bà)...
###- Thư ký phiên tòa: Ông (Bà)...  ** phần này được in đậm**(7)
###- Đại diện Viện kiểm sát nhân dân (8).....tham gia phiên tòa: ** phần này được in đậm**  
Ông (Bà)... Kiểm sát viên.

---
Trong các ngày........ tháng........ năm........(9) tại........................................... xét xử sơ thẩm
công khai(10) vụ án thụ lý số.........../...........(11)/TLST-HC ngày........ tháng........ năm........ về........................................(12) theo Quyết định đưa vụ án ra xét xử số ...../....../QĐXXST-HC ngày........
tháng........ năm........ giữa các đương sự: `Phần này lùi vào đầu dòng`
1. Người khởi kiện:(13)`Phần này lùi vào đầu dòng`
...`Phần này lùi vào đầu dòng`
Người đại diện hợp pháp của người khởi kiện:(14)`Phần này lùi vào đầu dòng`
...`Phần này lùi vào đầu dòng`
Người bảo vệ quyền và lợi ích hợp pháp của người khởi kiện:(15)`Phần này lùi vào đầu dòng`
...`Phần này lùi vào đầu dòng`
2. Người bị kiện: (16)`Phần này lùi vào đầu dòng`
... `Phần này lùi vào đầu dòng`
Người đại diện hợp pháp của người bị kiện:(17)`Phần này lùi vào đầu dòng`
... `Phần này lùi vào đầu dòng`
Người bảo vệ quyền và lợi ích hợp pháp của người bị kiện:(18)`Phần này lùi vào đầu dòng`
... `Phần này lùi vào đầu dòng`
3. Người có quyền lợi, nghĩa vụ liên quan:(19)`Phần này lùi vào đầu dòng`
... `Phần này lùi vào đầu dòng`
Người đại diện hợp pháp của người có quyền lợi, nghĩa vụ liên quan:(20)`Phần này lùi vào đầu dòng`
... `Phần này lùi vào đầu dòng`
Người bảo vệ quyền và lợi ích hợp pháp của người có quyền lợi, nghĩa vụ liên`Phần này lùi vào đầu dòng`
quan:(21)`Phần này lùi vào đầu dòng`
...`Phần này lùi vào đầu dòng`
4. Người làm chứng: (22)`Phần này lùi vào đầu dòng`
...`Phần này lùi vào đầu dòng`
5. Người giám định:(23)`Phần này lùi vào đầu dòng`
...`Phần này lùi vào đầu dòng`
6. Người phiên dịch:(24)`Phần này lùi vào đầu dòng`
... `Phần này lùi vào đầu dòng`

***NỘI DUNG VỤ ÁN (25) phần này in đậm, căn giữa***
...
...

***NHẬN ĐỊNH CỦA TÒA ÁN: (26) phần này in đậm, căn giữa***
[1]...
[2]...

Vì các lẽ trên,

***QUYẾT ĐỊNH: phần này in đậm, căn giữa***
Căn cứ vào.... (27)
.... (28).....
................
............(29)

***Nơi nhận: phần này in đậm căn lề trái**
(Ghi những nơi mà Tòa án cấp sơ thẩm phải giao hoặc gửi bản án theo quy định tại Điều 196 của Luật TTHC) phần này căn lề trái.


***TM. HỘI ĐỒNG XÉT XỬ SƠ THẨM nội dung phần này in đậm, nằm ở bên phải, không liên quan đến các class hoặc id trước**
THẨM PHÁN - CHỦ TỌA PHIÊN TÒA nội dung phần này in đậm , nằm ở bên phải, không liên quan đến các class hoặc id trước**
(Ký tên, ghi rõ họ tên, đóng dấu) nội dung phần này in nghiên, nằm ở bên phải, nội dung căn giữa, không liên quan đến các class hoặc id trước



**Yêu cầu:**  
- Bỏ phần giới thiệu dài dòng khúc đầu của hệ thống, hãy tập trung vào phần soạn bản án.
- Dùng thông tin tôi cung cấp để soạn thảo hoàn chỉnh bản án, bằng cách điền vào các chỗ trống, {case_details}.
- Căn cứ, viện dẫn điều luật tại Việt Nam một cách chính xác, dựa trên thông tin tình huống vụ án mà tôi cung cấp để hoàn thành soạn thảo bản án.
- Viết đúng định dạng bản án theo văn phong pháp lý, trang trọng, khách quan.  
- Không viết gộp đoạn, hãy chia từng phần theo đúng nhãn tiêu đề như trong mẫu.
- Định dạng kết quả trả về trong khổ giấy A4.
- Bắt buộc trả về kết quả dùng html5, và định dạng css chuẩn mực, không cần css cho thẻ <body>, phù hợp với trang A4 để dễ dàng hiển thị và in ấn, không tự ý css ngoài những gì mà tôi cung cấp dưới đây. Cụ thể như sau: 
<style>

        .header {{
            display: flex  !important;
            justify-content: space-between  !important;
            margin-bottom: 20px  !important;
        }}

        .header-left {{
            text-align: left  !important;
        }}

        .header-right {{
            text-align: center  !important;
        }}
        .header p{{
             margin: 0  !important;

        }}
        .indented {{
            margin-left: 20px  !important;
        }}

        .center-bold {{
            font-weight: bold  !important;
            text-align: center  !important;
        }}

        .italic {{
            font-style: italic  !important;
        }}
         / Print-specific styles /
			@media print {{
				@page {{
					size: A4  !important;
					margin: 15mm  !important; / Standard margins for A4 printing /
				}}
				body {{
					margin: 0  !important;
					padding: 0  !important;
					font-size: 12pt  !important; / Standard for printed documents /
					line-height: 1.5  !important;
					text-align: justify  !important;
					color: #000  !important;
				}}
                .header{{
                display: block  !important;
                }}
			}}
    </style>
"""

        # Call Gemini to generate the judgment
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
        judgment = gen.get("text", "Không có phản hồi từ mô hình.")
    except Exception as e:
        logger.exception("Error drafting judgment")
        return jsonify({"error": "Lỗi khi soạn thảo bản án: " + str(e)}), 500

    return jsonify({
        "judgment": judgment,
        "top_banan_documents": top_banan_docs,
        "top_anle_documents": top_anle_docs
    })
    
@app.route("/")
def home():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

