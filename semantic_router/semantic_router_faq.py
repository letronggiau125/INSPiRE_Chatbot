from semantic_router.routers.semantic import SemanticRouter
from semantic_router.routers.route import Route
from semantic_router.router_sample import (
    muon_tra_sample, quydinh_sample, huongdan_sample,
    dichvu_sample, lienhe_sample, chitchat_sample, chitchat_responses
)

# 🛠️ 1️⃣ Kiểm tra dữ liệu FAQ trước khi khởi tạo Router
faq_samples = {
    "faq_muon_tra": muon_tra_sample,
    "faq_quydinh": quydinh_sample,
    "faq_huongdan": huongdan_sample,
    "faq_dichvu": dichvu_sample,
    "faq_lienhe": lienhe_sample,
    "chitchat": chitchat_sample
}

print("📋 Đang kiểm tra dữ liệu FAQ:")
for category, utterances in faq_samples.items():
    if not utterances:
        print(f"❌ Cảnh báo: {category} không có câu hỏi nào!")
    else:
        print(f"✅ {category}: {len(utterances)} câu hỏi được nạp.")

# 🏗️ 2️⃣ Tạo các danh mục FAQ với mẫu từ `router_sample.py`
muon_tra_route = Route(name="faq_muon_tra", utterances=muon_tra_sample)
quydinh_route = Route(name="faq_quydinh", utterances=quydinh_sample)
huongdan_route = Route(name="faq_huongdan", utterances=huongdan_sample)
dichvu_route = Route(name="faq_dichvu", utterances=dichvu_sample)
lienhe_route = Route(name="faq_lienhe", utterances=lienhe_sample)
chitchat_route = Route(name="chitchat", utterances=chitchat_sample)

# 🚀 3️⃣ Khởi tạo SemanticRouter
semanticRouter = SemanticRouter(routes=[
    muon_tra_route, quydinh_route, huongdan_route, 
    dichvu_route, lienhe_route, chitchat_route
])

# 🤖 4️⃣ Hàm tìm kiếm và phản hồi
def get_response(query):
    category = semanticRouter.guide(query)[1]
    
    if category == "chitchat":
        return chitchat_responses.get(query, "Tôi không hiểu câu hỏi của bạn, nhưng tôi sẵn sàng giúp đỡ!")
    
    return f"🔹 Câu hỏi thuộc danh mục: {category}. Tìm kiếm trong cơ sở dữ liệu..."
