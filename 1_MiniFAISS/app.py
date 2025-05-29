from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 加载数据
with open('sentences.txt', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

# 2. 加载模型
print("正在加载模型...")
model = SentenceTransformer('moka-ai/m3e-small')

# 3. 编码
print("正在生成向量...")
embeddings = model.encode(sentences, convert_to_tensor=False)

# 4. 建立FAISS索引
embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

# 5. 用户输入查询
while True:
    query = input("\n请输入一句话（输入 'exit' 退出）：")
    if query.lower() == 'exit':
        break

    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k=3)

    print("最相近的句子：")
    for idx in I[0]:
        print(f"- {sentences[idx]}")
