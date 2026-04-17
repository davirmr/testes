import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1️⃣ FUNÇÃO: Carrega os textos dos arquivos JSON
def carregar_textos(caminhos_arquivos):
    textos = []
    for caminho in caminhos_arquivos:
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)  # Converte JSON para lista de dicionários
            for item in dados:
                item_limpo = {k.strip(): v for k, v in item.items()}  # Remove espaços das chaves
                if 'text' in item_limpo and item_limpo['text'].strip():
                    textos.append(item_limpo['text'].strip())  # Adiciona texto limpo à lista
    return textos

# 2️⃣ FUNÇÃO: Gera embeddings e cria a Vector Store (FAISS)
def preparar_busca(textos):
    modelo = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = modelo.encode(textos, show_progress_bar=False)  # Transforma textos em vetores
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normaliza vetores

    dimensao = embeddings.shape[1]  # Pega dimensão dos vetores
    index_faiss = faiss.IndexFlatIP(dimensao)  # Cria índice FAISS (Produto Interno = Cosseno)
    index_faiss.add(embeddings)  # Armazena embeddings na vector store

    return modelo, index_faiss, textos

# 3️⃣ FUNÇÃO: Busca semântica interativa
def buscar(query, modelo, index_faiss, textos, top_k=3):
    vec_query = modelo.encode([query])  # Gera embedding da consulta
    vec_query = vec_query / np.linalg.norm(vec_query, axis=1, keepdims=True)  # Normaliza

    # Busca os top_k vizinhos mais próximos no índice
    distancias, indices = index_faiss.search(vec_query, top_k)

    resultados = []
    for dist, idx in zip(distancias[0], indices[0]):
        resultados.append({'score': float(dist), 'texto': textos[idx]})
    return resultados

# 🚀 EXECUÇÃO INTERATIVA
if __name__ == "__main__":
    arquivos = ['noticias-govbr1.json', 'noticias-govbr2.json']
    
    print("📥 Carregando documentos...")
    docs = carregar_textos(arquivos)
    print(f"✅ {len(docs)} notícias carregadas.\n")

    print("🧠 Gerando embeddings e indexando no FAISS...")
    modelo_emb, index, lista_docs = preparar_busca(docs)
    print("✅ Vector Store pronta. Pronto para busca!\n")

    print("💡 Digite sua consulta e pressione ENTER. Digite 'sair' para encerrar.")
    while True:
        try:
            query = input("\n📝 Consulta: ").strip()  # Lê entrada do usuário
            
            if not query:  # Ignora linhas vazias
                continue
            if query.lower() in ['sair', 'exit', 'quit']:  # Permite saída amigável
                print("👋 Encerrando sistema. Até logo!")
                break

            print(f"\n🔎 Buscando por: '{query}'")
            res = buscar(query, modelo_emb, index, lista_docs, top_k=3)
            
            # Filtra resultados muito distantes (score < 0.3 geralmente indica baixa relevância)
            res_relevantes = [r for r in res if r['score'] > 0.15]
            
            if not res_relevantes:
                print("📭 Nenhum documento semanticamente relevante encontrado.")
            else:
                for i, r in enumerate(res_relevantes, 1):
                    # Exibe score e os 340 primeiros caracteres do texto
                    print(f"  {i}. [Similaridade: {r['score']:.3f}] {r['texto'][:340]}...")
            print("-" * 60)
            
        except KeyboardInterrupt:  # Permite sair com Ctrl+C
            print("\n👋 Encerrando sistema. Até logo!")
            break