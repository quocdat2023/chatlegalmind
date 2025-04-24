from flask import Flask, request, jsonify, render_template,make_response, redirect, url_for, session
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bcrypt
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain.prompts import ChatPromptTemplate
import os
from typing import List
import google.generativeai as genai
from datetime import datetime

genai.configure(api_key="AIzaSyA2RKWDRHuSVm8X5ez30-5NWbF0F4QdJGo")

app = Flask(__name__)
CORS(app)  # Cho phép xử lý CORS nếu cần

app.secret_key = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/your_database_name'
mongo = PyMongo(app)

genai.configure(api_key="AIzaSyA2RKWDRHuSVm8X5ez30-5NWbF0F4QdJGo")
sys.stdout.reconfigure(encoding='utf-8')

# Bước 1: Load mô hình tạo embedding
embedding_model = SentenceTransformer('hiieu/halong_embedding')  # Bạn có thể thay đổi mô hình khác nếu cần
# Tải lại FAISS index từ tệp
loaded_faiss_index = faiss.read_index("source/faiss_index.index")

# Đường dẫn đến tệp JSON đã lưu
input_path = "source/metadata.json"

# Tải dữ liệu từ tệp JSON
with open(input_path, "r", encoding="utf-8") as f:
    chunks_with_metadata = json.load(f)

# Đọc file JSON
with open('source/graph_anle.json', 'r', encoding='utf-8') as f:
    graph_json = json.load(f)

# Bước 4: Hàm truy vấn
def search(query, top_k=5):
    # Chuyển câu truy vấn thành embedding
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    # Tìm kiếm top_k kết quả
    distances, indices = loaded_faiss_index.search(query_embedding, top_k)
    # Trả về kết quả
    results = [{"text": chunks_with_metadata[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return results


# Bước 1: Load mô hình Re-Ranking (RARank)
rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Mô hình Re-Ranking
tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)

def rerank(query, search_results):
    scores = []
    for result in search_results:
        # Đảm bảo text là chuỗi
        text = str(result["text"])  # Chuyển đổi sang chuỗi nếu cần thiết

        # Ghép truy vấn và kết quả thành một input
        inputs = tokenizer.encode_plus(query, text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            score = rerank_model(**inputs).logits.item()  # Điểm số liên quan
        scores.append(score)

    # Thêm điểm số vào kết quả và sắp xếp lại
    for i, result in enumerate(search_results):
        result["rerank_score"] = scores[i]

    return sorted(search_results, key=lambda x: x["rerank_score"], reverse=True)


def search_with_rerank(query, top_k=5):
    # Tìm kiếm top_k từ FAISS
    try:
        search_results = search(query, top_k=top_k)
        if not search_results:
            raise ValueError("No search results returned.")
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    # Áp dụng Re-Ranking
    try:
        reranked_results = rerank(query, search_results)
        if not reranked_results:
            raise ValueError("No re-ranked results returned.")
    except Exception as e:
        print(f"Error during re-ranking: {e}")
        return []

    return reranked_results


# Hàm chỉnh sửa prompt_template
def prompt_template(docs: List[dict], original_query: str) -> str:
   # Kiểm tra các tài liệu có chứa khóa 'text', 'source', 'page' và chuyển mọi giá trị thành chuỗi
    context = "\n".join([str(doc.get('text', {}).get('text', 'Thông tin không có sẵn')) for doc in docs])
    page = "\n".join([str(doc.get('text', {}).get('page', 'Thông tin không có sẵn')) for doc in docs])
    source = "\n".join([str(doc.get('text', {}).get('source', 'Thông tin không có sẵn'))
                    .replace('/content/drive/MyDrive/data_law/', '')
                    .replace('.pdf', '') for doc in docs])

    # Đảm bảo câu hỏi được bao quanh bởi dấu ngoặc đơn
    original_query = f"'{original_query}'"

    response_prompt = f"""
    `Bạn là một chuyên gia tư vấn pháp luật với hơn 30 năm kinh nghiệm trong mọi lĩnh vực pháp lý tại Việt Nam. Nhiệm vụ của bạn là phân tích và trả lời các câu hỏi về án lệ pháp luật Việt Nam theo hình thức "Án lệ tương đồng và lập luận pháp lý". Bạn phải đảm bảo câu trả lời chính xác, dễ hiểu và tuân thủ các quy định pháp luật hiện hành.

    Dưới đây là câu hỏi bạn cần trả lời:
    {original_query}

    **Thông tin tham khảo để trả lời**:
    Nội dung án lệ: {context}
    Nguồn án lệ: {source}

    **Hướng dẫn trả lời**:

    1. **Cấu trúc câu trả lời**:  

    - **Giới thiệu**: Đưa ra án lệ tương đồng với tình huống cần giải quyết.  
    - **Nội dung án lệ**: Mô tả nội dung chính của án lệ liên quan.  
    - **Áp dụng vào tình huống**: Phân tích tình huống cụ thể, chỉ ra các yếu tố pháp lý có liên quan.  
    - **Lập luận pháp lý**: Giải thích cơ sở pháp lý, viện dẫn điều luật và công văn hướng dẫn (nếu có).  
    - **Kết luận**: Xác định tội danh hoặc hướng xử lý phù hợp theo pháp luật. 

    2. **Ví dụ về câu trả lời mong muốn:
    - Có 2 trường hợp, thứ nhất là có 1 án lệ thì trình như sau:
        Án lệ số 47/2021/AL, trang số 2  
            1. Nội dung: Xác định tội danh khi bị cáo dùng hung khí nguy hiểm (dao) tấn công vào vùng trọng yếu (tim, gan, não...) của nạn nhân (trang số 5,6,7), dù nạn nhân không chết nhưng hành vi thể hiện ý định cố ý giết người.  
            2. Áp dụng vào tình huống:  
            3. Hành vi: Nam thanh niên dùng dao đâm vào vùng trọng yếu (ví dụ: bụng, ngực), gây thương tích 13% cơ thể (tương tự vụ án trong Án lệ 47).  
            4. Ý thức chủ quan: Nếu chứng minh được bị cáo có chủ đích nhắm vào vùng nguy hiểm (trang số 1,4,7), tội danh sẽ là "Giết người chưa đạt" (Điều 123 Bộ luật Hình sự 2015).  
            5. Lập luận pháp lý: Công văn 100/TANDTC-PC (2023) quy định án lệ 47 chỉ áp dụng nếu hành vi thể hiện sự "côn đồ, hung hãn, coi thường tính mạng người khác".  
    - Trường hợp thứ hai có nhiều án lệ thì trình bày như sau:
        A. Án lệ số 47/2021/AL, trang số 2  
            1. Nội dung: Xác định tội danh khi bị cáo dùng hung khí nguy hiểm (dao) tấn công vào vùng trọng yếu (tim, gan, não...) của nạn nhân (trang số 5,6,7), dù nạn nhân không chết nhưng hành vi thể hiện ý định cố ý giết người.  
            2. Áp dụng vào tình huống:  
            3. Hành vi: Nam thanh niên dùng dao đâm vào vùng trọng yếu (ví dụ: bụng, ngực), gây thương tích 13% cơ thể (tương tự vụ án trong Án lệ 47).  
            4. Ý thức chủ quan: Nếu chứng minh được bị cáo có chủ đích nhắm vào vùng nguy hiểm (trang số 1,4,7), tội danh sẽ là "Giết người chưa đạt" (Điều 123 Bộ luật Hình sự 2015).  
            5. Lập luận pháp lý: Công văn 100/TANDTC-PC (2023) quy định án lệ 47 chỉ áp dụng nếu hành vi thể hiện sự "côn đồ, hung hãn, coi thường tính mạng người khác".  

        B. Án lệ số 17/2018/AL, trang số 5
            1. Nội dung: Xác định tình tiết "có tính chất côn đồ" trong tội giết người khi có đồng phạm.  
            2. Áp dụng vào tình huống:  
            3. Dù vụ án chỉ có một nghi phạm, nhưng hành vi dùng dao tấn công sau khi cãi vã có thể bị xem là hung hãn, côn đồ nếu bị cáo chủ động tấn công mà không có yếu tố phòng vệ.  

        C. Án lệ số 28/2019/AL, trang số 7
            1. Nội dung: Xử lý tội "giết người trong trạng thái tinh thần bị kích động mạnh".  
            2. Áp dụng vào tình huống:  
            3. Nếu cuộc cãi vã dẫn đến kích động mạnh, tòa có thể xem xét giảm nhẹ hình phạt. Tuy nhiên, **việc mang theo dao sẵn** có thể phủ nhận yếu tố này vì hành vi có tính toán trước.  
    - Chú ý thêm số thứ tự la mã trước các nội dung trong câu trẳ lời như sau:
        I. Giới thiệu:
        II. Nội dung án lệ:
        III. Áp dụng vào tình huống:
        IV. Lập luận pháp lý:
        V. Kết luận:

    3. Tuân thủ nội dung:
    - Chỉ sử dụng thông tin từ nội dung được cung cấp, có thể suy diễn và suy luận. Nhưng không thêm thông tin từ bên ngoài.
    - Không cần giới thiệu bạn là ai ở phần trả lời ban đầu, chú ý bạn là hệ thống truy vấn, tư vấn án lệ thông minh, dùng kiến trúc RAG.
    - Có trích dẫn trang số bao nhiêu từ án lệ nào đang truy vấn từ {context}

    4. Xử lý trường hợp không có thông tin:
    - Nếu không có thông tin phù hợp trong dữ liệu cung cấp, trả lời như sau: *'Xin lỗi, câu hỏi này không nằm trong phần kiến thức của tôi.'*
    - Không dùng từ "giả định, ví dụ" hoặc những cụm từ ko chắc chắn.
    - Không tự giới thiệu mình là ai, nếu có câu hỏi này thì chỉ cần trả lời **Tôi là một chuyên gia tư vấn pháp luật thông minh. Nhiệm vụ của tôi là phân tích và trả lời các câu hỏi về án lệ pháp luật Việt Nam theo hình thức "Án lệ tương đồng và lập luận pháp lý".**
    - Trường hợp kết quả không nằm trong kiến thức,có thể trả lời bằng cách dùng {original_query} để tham khảo trên internet và trả về kết quả sau khi tham khảo, có trích dẫn nguồn, trang đầy đủ. Không cần xin lỗi, tập trung vào câu trả lời chính.
    
    5. **Phong cách trình bày**:
    - Sử dụng ngôn ngữ pháp lý chính xác, chuyên nghiệp.  
    - Trình bày rõ ràng, logic, có đánh số hoặc dấu đầu dòng để dễ đọc
    - Trích dẫn cụ thể điều luật, công văn hướng dẫn để tăng tính thuyết phục. 
    - Ngắn gọn, rõ ràng, súc tích nhưng vẫn đảm bảo đủ ý.
    - Chuyên nghiệp và chính xác theo ngôn ngữ pháp luật Việt Nam.
    - Tránh sử dụng ngôn ngữ không trang trọng hoặc diễn đạt quá phức tạp.

    7. Đảm bảo chất lượng:
        - Suy luận, diễn giải đối chiếu kiến thức đã biết về pháp luật Việt Nam, trước khi đưa ra câu trả lời cuối cùng.
        - Mọi câu trả lời cần được kiểm tra tính chính xác và rõ ràng trước khi gửi đi.
        - Có trích dẫn trang số bao nhiêu từ án lệ nào truy vấn.
        - Kết thúc câu trả lời bằng một tóm lược ngắn gọn nếu cần thiết. 
        - Biết rằng trang án lệ được xác định ở đầu trang, có ký tự \n và số trang, ví dụ \n2 là trang 2, 
        \n3 là trang 3, \n3\n4 là trang số 3

    8. Định dạng câu trả lời:
    Câu trả lời cần được trình bày dưới dạng rõ ràng, có cấu trúc, với các phần được phân chia rõ ràng,in đậm những keyword chính bằng thẻ  và in nghiên keyword phụ bằng thẻ <i></i>, bao gồm:
        - Giới thiệu: Tóm tắt ngắn gọn vấn đề.
        - Nội dung chính: Chi tiết câu trả lời với các dẫn chứng pháp lý.
        - Kết luận: Tóm tắt kết quả và hướng xử lý.

    **Lưu ý:**
    - Không cần giới thiệu 30 năm kinh nghiệm.
    - Khi gặp câu hỏi bạn là ai, who are you, câu trả lời như sau:  
    *"Xin chào, tôi là hệ thống truy vấn án lệ pháp luật Việt Nam, có tên là Legalmind!"*  

    """

    return response_prompt


@app.route("/query", methods=["POST"])
def query():
    try:
        # Lấy dữ liệu JSON từ request
        data = request.json
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "Câu hỏi không hợp lệ!"}), 400

        # Tìm kiếm tài liệu có liên quan
        docs = search_with_rerank(query=question, top_k=5)

        # Nếu không có tài liệu nào được tìm thấy
        if not docs:
            return jsonify({"error": "Không tìm thấy tài liệu phù hợp!"}), 404

        # Kết hợp các câu trả lời từ search với rerank
        combined_answers = []
        for doc in docs:
        
            context = doc.get('text', {}).get('text', 'Không có thông tin phù hợp.')
            page = doc.get('text', {}).get('page', 'Thông tin không có sẵn')
            source = doc.get('text', {}).get('source', 'Thông tin không có sẵn').replace('/content/drive/MyDrive/data_law/', '').replace('.pdf', '')
            # Cập nhật câu trả lời với nguồn và trang án lệ tương ứng
            combined_answers.append(f"**Nội dung án lệ:** {context}\n**Nguồn án lệ:** {source}\n**Trang án lệ:** {page}")

        # Tạo chuỗi phản hồi từ template
        combined_answer_text = "\n\n".join(combined_answers)
        response_chain = prompt_template(docs, question) 
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(response_chain)
        # Create the model
        generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536,
        "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        generation_config=generation_config,
        )

        chat_session = model.start_chat(
        history=[
        ]
        )

        responses = chat_session.send_message(response_chain)
        res =  make_response(jsonify({
            "final_response": responses.text,
            "source": "\n".join([str(doc.get('text', {}).get('source', 'Thông tin không có sẵn')) for doc in docs])
        }))
        return res

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Đã có lỗi xảy ra trong quá trình xử lý câu hỏi!"}), 500

# @app.route("/query", methods=["POST"])
# def query():
#     try:
#         # Lấy dữ liệu JSON từ request
#         data = request.json
#         question = data.get("question", "")

#         if not question:
#             return jsonify({"error": "Câu hỏi không hợp lệ!"}), 400

#         # Khởi tạo ChatGoogleGenerativeAI với các tham số cần thiết
#         response_model = ChatGoogleGenerativeAI(
#             google_api_key="AIzaSyCeCV7D4JkDnzY6iYgYrMt5YKJnuhX5p_4",  # API key Google
#             model="gemini-2.0-flash-thinking-exp-01-21",  # Mô hình Gemini
#             temperature=0.2,
#             max_tokens=3000,
#             top_p=0.6,
#         )

#         # Tìm kiếm tài liệu có liên quan
#         docs = search_with_rerank(query=question, top_k=5)
#         source = "\n".join([str(doc.get('text', {}).get('source', 'Thông tin không có sẵn')) for doc in docs])

#         # Tạo chuỗi phản hồi từ template
#         response_chain = prompt_template(docs, question)

#         # Gọi mô hình để nhận câu trả lời
#         final_response = response_model.invoke(response_chain)

#         # Lưu vào cookie (giới hạn kích thước cookies, lưu trữ tối đa 4KB)
#         response = make_response(jsonify({
#             "final_response": final_response.content,
#             "source": source
#         }))
#         return response

#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({"error": "Đã có lỗi xảy ra trong quá trình xử lý câu hỏi!"}), 500


# @app.route("/query", methods=["POST"])
# def query():
#     data = request.json
#     question = data.get("question", "")

#     # Khởi tạo ChatGoogleGenerativeAI với các tham số cần thiết
#     response_model = ChatGoogleGenerativeAI(
#         google_api_key="AIzaSyCeCV7D4JkDnzY6iYgYrMt5YKJnuhX5p_4",  # Cung cấp Google API key của bạn
#         model="gemini-2.0-flash-thinking-exp-01-21",  # Mô hình Gemini hoặc mô hình bạn muốn sử dụng
#         temperature=0.2,
#         max_tokens=3000,
#         top_p=0.6,
#     )

#     docs = search_with_rerank(query=question, top_k=5)  # Giả sử đây là hàm tìm kiếm tài liệu
#     source = "\n".join([str(doc.get('text', {}).get('source', 'Thông tin không có sẵn')) for doc in docs])
#     # Tạo chuỗi phản hồi từ template
#     response_chain = prompt_template(docs, question)

#     # Gọi mô hình để nhận câu trả lời và trả về kết quả
#     final_response = response_model.invoke(response_chain)  # Không cần .strip() nữa

#     # Trả về văn bản từ đối tượng phản hồi
#     response = {"final_response": final_response.content, "source": source}
#     return response

@app.route("/")
def index():
    current_time = datetime.now().strftime("%I:%M:%S %p")
    return render_template("index.html", time=current_time)

# @app.route("/query", methods=["POST"])
# def query():
#     # Parse the request data
#     data = request.json
#     question = data.get("question", "")

#     search_results = search_with_rerank(question, top_k=5)  # Call your `search` function

#         # Extract required details from search results
#     top_answers = [result["text"]["text"] for result in search_results]
#     similarity_scores = [float(result["distance"]) for result in search_results]
#     rerank_scores = [float(result["rerank_score"]) for result in search_results]
#     sources = [result["text"]["source"] for result in search_results]
#     pages = [result["text"]["page"] for result in search_results]
#     source_paths = [source.replace("/content/drive/MyDrive/data_law/", "") for source in sources]
#         # Construct the response
#     response = {
#             "question": question,
#             "top_answers": top_answers,
#             "similarity_scores": similarity_scores,
#             "rerank_scores": rerank_scores,
#             "sources": source_paths,
#             "pages": pages,
#         }
#     return jsonify(response)


# @app.route("/get_case_data", methods=["POST"])
# def get_case_data():
#     data = request.get_json()
#     case_number = data.get("case_number")

#     # Find the target case
#     target_case = next((case for case in graph_json if case["case_number"] == case_number), None)
#     if not target_case:
#         return jsonify({"error": "Case not found"}), 404

#     # Build the graph structure
#     graph_data = {
#         "nodes": [{"id": target_case["case_number"], "name": target_case["case_number"]}],
#         "links": []
#     }

#     for related_case in target_case.get("case_relative_details", []):
#         graph_data["nodes"].append({"id": related_case["case_number"], "name": related_case["case_number"]})
#         graph_data["links"].append({"source": target_case["case_number"], "target": related_case["case_number"]})

#     return jsonify(graph_data)

@app.route("/get_case_data", methods=["GET"])
def get_case_data():
    case_number = request.args.get("case_number")

    # Check if the case number is provided
    if not case_number:
        return jsonify({"error": "Case number is required"}), 400

    # Find the target case
    target_case = next((case for case in graph_json if case["case_number"] == case_number), None)
    if not target_case:
        return jsonify({"error": "Case not found"}), 404

    # Build the graph structure
    graph_data = {
        "nodes": [{"id": target_case["case_number"], "name": target_case["case_number"]}],
        "links": []
    }

    # Add related cases as nodes and create links between the target case and related cases
    for related_case in target_case.get("case_relative_details", []):
        graph_data["nodes"].append({"id": related_case["case_number"], "name": related_case["case_number"]})
        graph_data["links"].append({"source": target_case["case_number"], "target": related_case["case_number"]})

    return jsonify({"graph_data": graph_data})

@app.route("/graphs")
def graphs():
    case_number = request.args.get("case_number")
    if not case_number:
        return "Case number is required", 400

    return render_template("graph.html", case_number=case_number)  # Render graph page for specific case number

if __name__ == "__main__":
    app.run(debug=True)
