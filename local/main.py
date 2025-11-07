import os
import json
import chromadb
from chromadb.utils import embedding_functions
import ollama
import time
import threading
import shutil

# 确保目录存在
os.makedirs("memory", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# 初始化 ChromaDB
client = chromadb.PersistentClient(path="./memory")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(
    name="conversations",
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

# 对话计数与版本
dialogue_count = 0
version_file = "../models/version.txt"
current_version = "v0.1"
if os.path.exists(version_file):
    with open(version_file, "r") as f:
        current_version = f.read().strip()

def save_dialogue(user_input, response):
    global dialogue_count
    # 保存到 JSONL
    with open("dialogues.jsonl", "a", encoding="utf-8") as f:
        json.dump({"user": user_input, "agent": response, "timestamp": time.time()}, f, ensure_ascii=False)
        f.write("\n")
    
    # 保存到 ChromaDB
    collection.add(
        documents=[f"User: {user_input}\nAgent: {response}"],
        metadatas=[{"turn": dialogue_count}],
        ids=[f"turn_{dialogue_count}"]
    )
    dialogue_count += 1

def retrieve_memories(query, n=3):
    try:
        results = collection.query(query_texts=[query], n_results=n)
        return "\n".join(results["documents"][0]) if results["documents"][0] else ""
    except Exception as e:
        print(f"[Memory] 检索失败: {e}")
        return ""

def get_model_response(prompt, memories=""):
    context = f"记忆上下文（如有）:\n{memories}\n\n当前对话:" if memories else "当前对话:"
    full_prompt = f"{context}\nUser: {prompt}\nAgent:"
    
    try:
        # 尝试加载进化模型，若无则 fallback 到 phi3:mini
        model_name = "phi3:evolving"
        try:
            ollama.show(model_name)
        except:
            print("[Model] 首次运行，创建初始模型 phi3:evolving...")
            os.system("ollama create phi3:evolving -f ../Modelfile")
        return ollama.generate(model=model_name, prompt=full_prompt)["response"].strip()
    except Exception as e:
        print(f"[Ollama] 推理失败: {e}")
        return "抱歉，我暂时无法回答。"

def auto_upload_and_train():
    """自动上传对话并触发训练（模拟云端行为）"""
    print("[自动进化] 检测到10条新对话，正在自动上传并触发训练...")
    
    # 复制对话文件到云端目录（模拟上传）
    src = "dialogues.jsonl"
    dst = "../cloud/dialogues_uploaded.jsonl"
    shutil.copy2(src, dst)
    
    # 触发训练（在子线程中运行，避免阻塞聊天）
    def run_training():
        os.system("cd ../cloud && python train_lora.py")
        print("[自动进化] 训练完成！正在自动更新本地模型...")
        os.system("cd ../local && python update_model.py")
        print("[自动进化] 模型已更新，继续聊天吧！")
    
    training_thread = threading.Thread(target=run_training)
    training_thread.start()

def main():
    global dialogue_count
    # 加载已有对话数量
    if os.path.exists("dialogues.jsonl"):
        with open("dialogues.jsonl", "r", encoding="utf-8") as f:
            dialogue_count = sum(1 for _ in f)
    
    print(f"[启动] EvolvingAgent-Cloud {current_version} 启动成功！输入 'quit' 退出。")
    print("="*50)
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("再见！")
            break
        
        memories = retrieve_memories(user_input)
        response = get_model_response(user_input, memories)
        print(f"Agent: {response}")
        
        save_dialogue(user_input, response)
        
        # 每10条自动触发上传和训练
        if dialogue_count % 10 == 0 and dialogue_count > 0:
            auto_upload_and_train()

if __name__ == "__main__":
    main()