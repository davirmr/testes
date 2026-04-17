# -*- coding: utf-8 -*-
"""
Sistema de Busca Semântica com Vector Store (FAISS + Sentence Transformers)
Descrição: Implementação completa de busca semântica em documentos governamentais
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime

# Bibliotecas para embeddings e vector store
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize

class SemanticSearchSystem:
    """
    Sistema de busca semântica utilizando embeddings e FAISS
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Inicializa o sistema com um modelo de embeddings
        
        Args:
            model_name: Nome do modelo multilingual para português
        """
        print(f"🔧 Inicializando sistema de busca semântica...")
        print(f"📚 Carregando modelo: {model_name}")
        
        # Carregar modelo de embeddings (multilingual para suporte a português)
        self.model = SentenceTransformer(model_name)
        
        # Dimensão do embedding gerado pelo modelo
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Modelo carregado. Dimensão dos embeddings: {self.embedding_dim}")
        
        # Inicializar estruturas de dados
        self.documents = []
        self.embeddings = None
        self.index = None
        self.metadata = []
        
    def load_documents(self, json_files: List[str]):
        """
        Carrega documentos dos arquivos JSON
        
        Args:
            json_files: Lista de caminhos para arquivos JSON
        """
        print(f"\n📄 Carregando documentos de {len(json_files)} arquivos...")
        
        all_texts = []
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    for item in data:
                        # Criar documento estruturado
                        doc = {
                            'date': item.get('date', ''),
                            'title': item.get('title', 'Sem título'),
                            'text': item.get('text', ''),
                            'source_file': file_path
                        }
                        
                        # Adicionar apenas se houver texto
                        if doc['text']:
                            self.documents.append(doc)
                            all_texts.append(doc['text'])
                            
            except Exception as e:
                print(f"❌ Erro ao carregar {file_path}: {e}")
        
        print(f"✅ Carregados {len(self.documents)} documentos")
        return all_texts
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32):
        """
        Gera embeddings para os documentos
        
        Args:
            texts: Lista de textos para gerar embeddings
            batch_size: Tamanho do batch para processamento
        """
        print(f"\n🧠 Gerando embeddings para {len(texts)} documentos...")
        print(f"⏱️  Isso pode levar alguns segundos...")
        
        # Gerar embeddings em batches para otimização
        embeddings_list = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)
            
            if (i + batch_size) % 100 == 0:
                print(f"   Processados {i + batch_size} documentos...")
        
        # Concatenar todos os embeddings
        self.embeddings = np.vstack(embeddings_list)
        
        # Normalizar embeddings para similaridade de cosseno
        self.embeddings = normalize(self.embeddings, norm='l2')
        
        print(f"✅ Embeddings gerados. Shape: {self.embeddings.shape}")
        return self.embeddings
    
    def create_faiss_index(self):
        """
        Cria índice FAISS para busca eficiente
        """
        print(f"\n🏗️  Criando índice FAISS...")
        
        # Criar índice para similaridade de cosseno (inner product após normalização)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Adicionar embeddings ao índice
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ Índice FAISS criado com {self.index.ntotal} vetores")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Busca documentos semanticamente similares à consulta
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (documento, score de similaridade)
        """
        # Gerar embedding da consulta
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = normalize(query_embedding, norm='l2')
        
        # Buscar no índice FAISS
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Preparar resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def display_results(self, query: str, results: List[Tuple[Dict, float]]):
        """
        Exibe resultados da busca de forma formatada
        
        Args:
            query: Consulta original
            results: Resultados da busca
        """
        print("\n" + "="*80)
        print(f"🔍 CONSULTA: {query}")
        print("="*80)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n📌 RESULTADO {i} (Similaridade: {score:.3f})")
            print(f"📅 Data: {doc['date']}")
            print(f"📄 Texto: {doc['text'][:300]}...")
            print(f"📁 Fonte: {doc['source_file']}")
            print("-"*80)
    
    def save_index(self, path: str = "faiss_index.bin"):
        """
        Salva o índice FAISS em disco
        
        Args:
            path: Caminho para salvar o índice
        """
        if self.index is not None:
            faiss.write_index(self.index, path)
            print(f"💾 Índice salvo em: {path}")
    
    def load_index(self, path: str = "faiss_index.bin"):
        """
        Carrega índice FAISS do disco
        
        Args:
            path: Caminho do índice
        """
        self.index = faiss.read_index(path)
        print(f"📂 Índice carregado de: {path}")

def main():
    """
    Função principal demonstrando o sistema de busca semântica
    """
    print("🚀 INICIANDO SISTEMA DE BUSCA SEMÂNTICA")
    print("="*80)
    
    # 1. Criar instância do sistema
    search_system = SemanticSearchSystem()
    
    # 2. Carregar documentos
    json_files = ['noticias-govbr1.json', 'noticias-govbr2.json']
    texts = search_system.load_documents(json_files)
    
    # 3. Gerar embeddings
    embeddings = search_system.generate_embeddings(texts)
    
    # 4. Criar índice FAISS
    search_system.create_faiss_index()
    
    # 5. Salvar índice para uso futuro
    search_system.save_index()
    
    # 6. Demonstração de buscas semânticas
    print("\n" + "🎯 DEMONSTRAÇÃO DE BUSCAS SEMÂNTICAS".center(80))
    print("="*80)
    
    # Exemplo 1: Busca sobre políticas públicas
    query1 = "Quais são as principais políticas públicas para municípios brasileiros?"
    results1 = search_system.search(query1, top_k=2)
    search_system.display_results(query1, results1)
    
    # Exemplo 2: Busca sobre programas habitacionais
    query2 = "Programas de habitação popular e moradia para comunidades carentes"
    results2 = search_system.search(query2, top_k=2)
    search_system.display_results(query2, results2)
    
    # Exemplo 3: Busca sobre educação infantil
    query3 = "Medidas de proteção e desenvolvimento para crianças pequenas"
    results3 = search_system.search(query3, top_k=2)
    search_system.display_results(query3, results3)
    
    # Exemplo 4: Busca sobre drogas e segurança
    query4 = "Discussões sobre legalização e criminalização do uso de entorpecentes"
    results4 = search_system.search(query4, top_k=2)
    search_system.display_results(query4, results4)
    
    # Exemplo 5: Busca sobre reforma econômica
    query5 = "Mudanças no sistema tributário e controle da inflação"
    results5 = search_system.search(query5, top_k=2)
    search_system.display_results(query5, results5)
    
    # Estatísticas finais
    print("\n" + "📊 ESTATÍSTICAS DO SISTEMA".center(80))
    print("="*80)
    print(f"Total de documentos indexados: {len(search_system.documents)}")
    print(f"Dimensão dos embeddings: {search_system.embedding_dim}")
    print(f"Tamanho do índice FAISS: {search_system.index.ntotal} vetores")
    print(f"Modelo utilizado: {search_system.model._modules['0'].auto_model.config.name_or_path}")

class DocumentAnalyzer:
    """
    Classe adicional para análise e visualização dos embeddings
    """
    
    @staticmethod
    def analyze_embeddings(embeddings: np.ndarray, documents: List[Dict]):
        """
        Análise estatística dos embeddings gerados
        
        Args:
            embeddings: Matriz de embeddings
            documents: Lista de documentos
        """
        print("\n📈 ANÁLISE DOS EMBEDDINGS")
        print("="*80)
        
        # Estatísticas básicas
        print(f"Média dos embeddings: {np.mean(embeddings):.4f}")
        print(f"Desvio padrão: {np.std(embeddings):.4f}")
        print(f"Min: {np.min(embeddings):.4f}")
        print(f"Max: {np.max(embeddings):.4f}")
        
        # Distribuição de tamanhos de documentos
        doc_lengths = [len(doc['text'].split()) for doc in documents]
        print(f"\n📊 Estatísticas dos documentos:")
        print(f"  - Média de palavras por documento: {np.mean(doc_lengths):.1f}")
        print(f"  - Mediana: {np.median(doc_lengths):.1f}")
        print(f"  - Maior documento: {max(doc_lengths)} palavras")
        print(f"  - Menor documento: {min(doc_lengths)} palavras")

if __name__ == "__main__":
    # Executar sistema principal
    main()
    
    # Demonstrar análise adicional
    print("\n" + "🔬 ANÁLISE AVANÇADA".center(80))
    print("="*80)
    
    # Recriar sistema para análise
    search_system = SemanticSearchSystem()
    texts = search_system.load_documents(['noticias-govbr1.json', 'noticias-govbr2.json'])
    embeddings = search_system.generate_embeddings(texts)
    
    # Análise dos embeddings
    DocumentAnalyzer.analyze_embeddings(embeddings, search_system.documents)
