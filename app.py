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
    # Validate and enrich metadata
    for i, meta in enumerate(metadata_dict["metadata"]):
        if "type" not in meta:
            meta["type"] = "banan"
        # Ensure required fields
        meta["case_summary"] = meta.get("case_summary", "No summary available")
        meta["legal_issues"] = meta.get("legal_issues", "No legal issues specified")
        meta["court_reasoning"] = meta.get("court_reasoning", "No reasoning provided")
        meta["decision"] = meta.get("decision", "No decision available")
        meta["relevant_laws"] = meta.get("relevant_laws", "No laws cited")
        # Warn if text is empty
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

def query_faiss_index(query: str, embeddings: SentenceTransformer, idx: faiss.Index, metadata: dict, k: int = 5, doc_type: str = None, max_threshold: float = 0.8):
    """Query FAISS index and return top-k similar documents, filtered by type and dynamic distance threshold."""
    query_emb = embeddings.encode([query], convert_to_numpy=True)
    distances, indices = idx.search(query_emb, k)
    results = []

    # Calculate dynamic threshold based on 50th percentile
    if len(distances[0]) > 0:
        distance_threshold = min(max_threshold, np.percentile(distances[0], 50))
    else:
        distance_threshold = max_threshold

    for dist, i in zip(distances[0], indices[0]):
        if 0 <= i < len(metadata["ids"]):
            result = {
                "id": metadata["ids"][i],
                "metadata": metadata["metadata"][i],
                "text": metadata["texts"][i],  # Return full text
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
        # Query judgments and precedents
        banan_results = query_faiss_index(
            question, embeddings, faiss_index, metadata_dict, k=5, doc_type="banan", max_threshold=0.8
        )
        banan_sum_results = query_faiss_index(
            question, embeddings, faiss_index, metadata_path_sum_dict, k=5, doc_type="banan", max_threshold=0.8
        )
        anle_results = query_faiss_index(
            question, embeddings, faiss_anle_index, anle_metadata_dict, k=5, doc_type="anle", max_threshold=0.8
        )

    except Exception as e:
        logger.exception("Error querying FAISS")
        return jsonify({"error": "Lỗi khi truy vấn FAISS: " + str(e)}), 500

    # Prepare judgment and precedent data
    top_banan_docs = [
        {
            "source": result["metadata"]["source"],
            "text": result["text"],
            "distance": result["distance"],
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
            "distance": result["distance"],
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
            "distance": result["distance"],
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

    # Prompt
    prompt = f"""
Bạn là chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. Bạn sẽ phân tích vấn đề pháp lý theo các bước chi tiết dưới đây để cung cấp câu trả lời chặt chẽ, rõ ràng, và thuyết phục, đảm bảo đề cập đến cả bản án và án lệ khi có sẵn.

**Câu hỏi:**  
{question}

**Thông tin tham khảo (bản án tương đồng):**  
{top_banan_docs if top_banan_docs else "Không tìm thấy bản án phù hợp. Phân tích dựa trên các quy định pháp luật hiện hành và nguyên tắc pháp lý chung."}

**Thông tin tham khảo (án lệ tương đồng):**  
{top_anle_docs if top_anle_docs else "Không tìm thấy án lệ phù hợp. Phân tích dựa trên các quy định pháp luật hiện hành và nguyên tắc pháp lý chung."}

**Hướng dẫn trả lời chi tiết:**

1. **Tổng quan về các bản án, án lệ tương đồng:**  
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
- Nếu xác định được bản án hay án lệ không phù hợp hãy bỏ quan, không đề cập đến trong câu trả lời.
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
Soạn thảo bản án cho một vụ án tại Việt Nam dựa trên thông tin vụ án và các tài liệu tham khảo sau. Bản án phải tuân thủ cấu trúc pháp lý chính thức, sử dụng ngôn ngữ pháp lý chính xác, chặt chẽ, và phù hợp với quy định pháp luật Việt Nam.

**Thông tin vụ án:**  
{case_details}

**Hướng dẫn soạn thảo bản án ngắn gọn nhất, đầy đủ ý, có tính chất pháp lý rõ ràng, là dàn ý gợi ý cho soạn thảo bản án, có đầy đủ chữ ký các bên liên quan. Lưu ý đây chỉ là hỗ trợ soạn thảo, nên không cần quá chi tiết: **

1. **Phần mở đầu:**  
   - Nêu rõ tên tòa án, số bản án, ngày tháng năm xét xử.  
   - Ghi thông tin các bên (nguyên đơn, bị đơn, người có quyền lợi và nghĩa vụ liên quan).  
   - Tóm tắt nội dung vụ án và yêu cầu khởi kiện.
   - Phải có đủ court header.

2. **Xét thấy (Phân tích sự kiện và căn cứ pháp lý):**  
   - Tóm tắt các sự kiện chính của vụ án dựa trên thông tin vụ án.  
   - Phân tích các vấn đề pháp lý trọng tâm, viện dẫn cụ thể các điều luật từ Bộ luật Dân sự, Luật Thương mại, hoặc các văn bản pháp luật liên quan.  
   - Nếu có bản án hoặc án lệ tương đồng, tham chiếu lập luận của tòa án, so sánh điểm tương đồng và khác biệt với vụ án hiện tại.  
   - Nếu không có bản án hoặc án lệ, phân tích dựa trên các nguyên tắc pháp lý và quy định pháp luật hiện hành.

3. **Nhận định của tòa án:**  
   - Đưa ra nhận định về trách nhiệm pháp lý, nghĩa vụ của các bên.  
   - Giải thích lý do chấp nhận hoặc bác bỏ các yêu cầu khởi kiện.  
   - Đảm bảo lập luận chặt chẽ, logic, và phù hợp với thực tiễn pháp lý Việt Nam.

4. **Quyết định (Phần tuyên án):**  
   - Nêu rõ quyết định của tòa án, bao gồm chấp nhận/bác bỏ yêu cầu khởi kiện, nghĩa vụ bồi thường (nếu có), và phân chia án phí.  
   - Viện dẫn điều luật cụ thể làm cơ sở cho quyết định.  

5. **Kết thúc:**  
   - Nêu quyền kháng cáo và thời hạn kháng cáo theo quy định pháp luật Việt Nam.

**Lưu ý quan trọng:**  
- Bắt buộc phải trả về đầy đủ, hoàn chỉnh cấu trúc một bản án, không được thiếu, chỉ ở mức gợi ý, tạo dàn ý cho người soạn thảo bản án, không cần chi tiết cụ thể, đầy đủ chữ ký các bên có liên quan.
- Phải có quốc hiệu nằm bên phải, tòa án, bản án nằm bên trái.
- Sử dụng ngôn ngữ pháp lý chính xác, trang trọng, và tuân thủ cấu trúc bản án theo quy định pháp luật Việt Nam.  
- Nếu bản án hoặc án lệ không phù hợp, không đề cập đến mà tập trung vào phân tích pháp lý dựa trên quy định pháp luật.  
- Đảm bảo bản án có tính thực tiễn, có thể áp dụng trực tiếp vào vụ án cụ thể.  
- Không sử dụng từ giả sử, ví dụ; không chào hỏi hoặc giới thiệu.  
- Nếu thông tin vụ án thiếu chi tiết, dựa vào các nguyên tắc pháp lý chung và quy định pháp luật hiện hành để soạn thảo.
- Nên tham khảo cách soạn thảo bản án của https://thuvienphapluat.vn/


Mẫu bản án đúng chuẩn tại Việt Nam phải tuân theo các quy định của pháp luật tố tụng hình sự và các văn bản hướng dẫn của Tòa án nhân dân tối cao. Hiện nay, mẫu bản án hình sự sơ thẩm được quy định tại Mẫu số 27-HS ban hành kèm theo Nghị quyết số 05/2017/NQ-HĐTP ngày 19 tháng 9 năm 2017 của Hội đồng Thẩm phán Tòa án nhân dân tối cao. 1    
1.
toaandaklak.gov.vn
toaandaklak.gov.vn

Dưới đây là cấu trúc chung của một bản án hình sự sơ thẩm đúng chuẩn, bạn có thể tham khảo:

TÒA ÁN NHÂN DÂN CẤP TỈNH/THÀNH PHỐ HOẶC CẤP HUYỆN/QUẬN

BẢN ÁN HÌNH SỰ SƠ THẨM

Số: .../..../HSST

Ngày ... tháng ... năm ...

TẠI PHIÊN TÒA CÔNG KHAI xét xử sơ thẩm vụ án hình sự thụ lý số: .../..../TLST-HS ngày ... tháng ... năm ... theo Quyết định đưa vụ án ra xét xử số: .../..../QĐXX-ST ngày ... tháng ... năm ... đối với bị cáo:

Họ và tên bị cáo: (Ghi đầy đủ họ tên, ngày, tháng, năm sinh, nơi sinh, quốc tịch, dân tộc, tôn giáo, nghề nghiệp, nơi cư trú, trình độ văn hóa, tiền án, tiền sự - nếu có)

Hội đồng xét xử gồm:

Thẩm phán - Chủ tọa phiên tòa: (Họ và tên Thẩm phán)
Các Thẩm phán: (Họ và tên các Thẩm phán khác)
Hội thẩm nhân dân: (Họ và tên các Hội thẩm nhân dân)
Thư ký phiên tòa: (Họ và tên Thư ký phiên tòa)

Đại diện Viện kiểm sát nhân dân: (Họ và tên Kiểm sát viên)

Người bào chữa cho bị cáo: (Nếu có, ghi rõ họ tên Luật sư, Văn phòng Luật sư/Công ty Luật)

Bị hại: (Ghi đầy đủ họ tên, ngày, tháng, năm sinh, nơi cư trú - nếu có)

Nguyên đơn dân sự: (Ghi đầy đủ họ tên, ngày, tháng, năm sinh, nơi cư trú - nếu có)

Bị đơn dân sự: (Ghi đầy đủ họ tên, địa chỉ - nếu có)

Người có quyền lợi, nghĩa vụ liên quan: (Ghi đầy đủ họ tên, địa chỉ - nếu có)

NỘI DUNG VỤ ÁN:

(Trình bày tóm tắt hành vi phạm tội của bị cáo theo Cáo trạng của Viện kiểm sát, lời khai của bị cáo tại phiên tòa, lời khai của bị hại, người làm chứng, kết quả giám định và các chứng cứ khác đã được thẩm tra tại phiên tòa.)

NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ:

(Phân tích, đánh giá các chứng cứ đã thu thập được, xác định tính chất, mức độ nguy hiểm cho xã hội của hành vi phạm tội, nhân thân của bị cáo, các tình tiết tăng nặng, giảm nhẹ trách nhiệm hình sự, xác định tội danh và điều khoản của Bộ luật Hình sự mà bị cáo phạm phải. Đánh giá về trách nhiệm dân sự và xử lý vật chứng (nếu có).)

QUYẾT ĐỊNH:

1. Về tội danh và hình phạt:

Tuyên bố bị cáo ... (họ và tên) phạm tội ... (tên tội danh) quy định tại khoản ... Điều ... của Bộ luật Hình sự.
Xử phạt bị cáo ... (họ và tên) ... (mức hình phạt chính, ví dụ: ... năm tù). Thời hạn chấp hành hình phạt tù tính từ ngày ... (ghi rõ ngày bắt tạm giam hoặc ngày bị cáo đến chấp hành án).
(Nếu có hình phạt bổ sung thì ghi rõ, ví dụ: Phạt tiền bị cáo ... đồng; Cấm bị cáo đảm nhiệm chức vụ ... trong thời hạn ... năm sau khi chấp hành xong hình phạt tù.)
2. Về trách nhiệm dân sự:

Buộc bị cáo ... (họ và tên) phải bồi thường cho ... (người được bồi thường) số tiền ... đồng.
(Các quyết định khác về trách nhiệm dân sự - nếu có.)
3. Về xử lý vật chứng:

(Nêu rõ quyết định xử lý đối với từng vật chứng của vụ án, ví dụ: Tịch thu sung quỹ nhà nước ...; Trả lại cho bị hại ...; Tiêu hủy ...)
4. Về án phí:

Bị cáo ... (họ và tên) phải chịu ... đồng án phí hình sự sơ thẩm.
(Quyết định về án phí dân sự sơ thẩm - nếu có.)
5. Về quyền kháng cáo:

Bị cáo, bị hại, nguyên đơn dân sự, bị đơn dân sự, người có quyền lợi, nghĩa vụ liên quan có quyền kháng cáo bản án này trong thời hạn 15 ngày kể từ ngày tuyên án.
Bản án được tuyên tại phiên tòa vào hồi ... giờ ... phút ngày ... tháng ... năm ...

TM. HỘI ĐỒNG XÉT XỬ

THẨM PHÁN - CHỦ TỌA PHIÊN TÒA

(Ký tên và đóng dấu)

Lưu ý quan trọng:

Đây chỉ là cấu trúc chung, nội dung chi tiết của từng phần sẽ thay đổi tùy thuộc vào từng vụ án cụ thể.
Các thông tin cần được điền đầy đủ, chính xác và phù hợp với diễn biến, chứng cứ của vụ án.
Việc viện dẫn điều luật phải chính xác theo quy định hiện hành của Bộ luật Hình sự và Bộ luật Tố tụng hình sự.
Bản án phải được viết rõ ràng, mạch lạc, có tính thuyết phục và đảm bảo tính pháp lý.

**Định dạng đầu ra:**  
- Trả về bản án dưới dạng văn bản thuần túy, có cấu trúc rõ ràng, phân chia các phần (Mở đầu, Xét thấy, Nhận định, Quyết định, Kết thúc).
- Có định dạng html5, bắt buộc dùng css có sẵn sau, không tự ý thêm cái khác:
<style>
        body {{
            font-family: "Jost", sans-serif;
        }}
        #center-align {{
            text-align: center;
        }}
        #bold-upper {{
            font-weight: bold;
            text-transform: uppercase;
        }}
        .section-title {{
             font-weight: bold;
             text-transform: uppercase;
             text-align: center;  
             margin-top: 20px;
             margin-bottom: 10px;
        }}
         .decision-section {{
              font-weight: bold;
              text-align: center;
              margin-top: 20px;
              margin-bottom: 10px;
         }}
         table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
         }}
         td, th {{
            padding: 5px;
            vertical-align: top;
         }}
         .party-info td:first-child {{
            width: 150px; / Adjust width as needed /
             font-weight: bold;
         }}
    </style>
để tiện cho việc hiển thị lên web.
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