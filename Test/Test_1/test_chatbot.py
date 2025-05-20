from reflection.reflection import Reflection

chatbot = Reflection(db_path="./chroma_db")

session_id = "test_user"
user_message = "Làm sao để gia hạn sách?"

response = chatbot.chat(session_id=session_id, user_message=user_message)
print(response)
