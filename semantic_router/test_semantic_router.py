import sys
import os

# Thêm thư mục gốc vào sys.path nếu chưa có
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("sys.path:", sys.path)

import unittest
from semantic_router.semantic_router_faq import semanticRouter, get_response
from router_sample import (
    muon_tra_sample, quydinh_sample, huongdan_sample,
    dichvu_sample, lienhe_sample, chitchat_sample, chitchat_responses
)

from semantic_router.semantic_router_faq import semanticRouter, get_response

class LibraryRouterTestCase(unittest.TestCase):
    def test_muon_tra_route(self):
        """Kiểm tra danh mục Mượn - Trả tài liệu."""
        for query in muon_tra_sample:
            print(f"📌 Kiểm tra: {query}")
            self.assertEqual(semanticRouter.guide(query)[1], "faq_muon_tra")

    def test_quydinh_route(self):
        """Kiểm tra danh mục Quy định thư viện."""
        for query in quydinh_sample:
            print(f"📌 Kiểm tra: {query}")
            self.assertEqual(semanticRouter.guide(query)[1], "faq_quydinh")

    def test_huongdan_route(self):
        """Kiểm tra danh mục Hướng dẫn sử dụng."""
        for query in huongdan_sample:
            print(f"📌 Kiểm tra: {query}")
            self.assertEqual(semanticRouter.guide(query)[1], "faq_huongdan")

    def test_dichvu_route(self):
        """Kiểm tra danh mục Dịch vụ thư viện."""
        for query in dichvu_sample:
            print(f"📌 Kiểm tra: {query}")
            self.assertEqual(semanticRouter.guide(query)[1], "faq_dichvu")

    def test_lienhe_route(self):
        """Kiểm tra danh mục Liên hệ - Hỗ trợ."""
        for query in lienhe_sample:
            print(f"📌 Kiểm tra: {query}")
            self.assertEqual(semanticRouter.guide(query)[1], "faq_lienhe")

    def test_chitchat_route(self):
        """Kiểm tra danh mục Chitchat."""
        for query in chitchat_sample:
            print(f"📌 Kiểm tra: {query}")
            self.assertEqual(semanticRouter.guide(query)[1], "chitchat")

    def test_chitchat_response(self):
        """Kiểm tra phản hồi của Chitchat."""
        for query in chitchat_sample:
            print(f"📌 Kiểm tra phản hồi: {query}")
            self.assertEqual(get_response(query), chitchat_responses.get(query))

# Chạy unittest
if __name__ == "__main__":
    unittest.main()
