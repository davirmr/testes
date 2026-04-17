# test_main.py
# Testes unitários para a API da Biblioteca Virtual

import pytest
from fastapi.testclient import TestClient
from main import app
import sqlite3
import os

# Criar cliente de teste
client = TestClient(app)

# Setup: limpar banco de dados antes dos testes
@pytest.fixture(autouse=True)
def setup_database():
    """Prepara o banco de dados para cada teste"""
    # Conectar ao banco
    conn = sqlite3.connect("biblioteca.db")
    cursor = conn.cursor()
    
    # Limpar todos os livros antes de cada teste
    cursor.execute("DELETE FROM livros")
    
    # Resetar o contador de ID
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='livros'")
    
    conn.commit()
    conn.close()
    
    yield  # Executa o teste
    
    # Após o teste, podemos limpar novamente (opcional)
    conn = sqlite3.connect("biblioteca.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM livros")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='livros'")
    conn.commit()
    conn.close()

# Teste 1: Cadastrar um novo livro
def test_cadastrar_livro():
    """Testa se o endpoint POST /livros cadastra corretamente"""
    response = client.post("/livros", json={
        "titulo": "Teste Livro",
        "autor": "Teste Autor",
        "data_publicacao": "2024-01-01",
        "resumo": "Este é um livro de teste com resumo bem completo."
    })
    
    # Verificar status code
    assert response.status_code == 201
    
    # Verificar dados retornados
    dados = response.json()
    assert dados["id"] == 1
    assert dados["titulo"] == "Teste Livro"
    assert dados["autor"] == "Teste Autor"
    assert dados["data_publicacao"] == "2024-01-01"
    assert dados["resumo"] == "Este é um livro de teste com resumo bem completo."

# Teste 2: Consultar todos os livros
def test_consultar_todos_livros():
    """Testa se GET /livros retorna todos os livros cadastrados"""
    # Cadastrar 2 livros primeiro
    client.post("/livros", json={
        "titulo": "Livro A",
        "autor": "Autor A",
        "data_publicacao": "2020-01-01",
        "resumo": "Resumo do livro A"
    })
    
    client.post("/livros", json={
        "titulo": "Livro B",
        "autor": "Autor B",
        "data_publicacao": "2021-01-01",
        "resumo": "Resumo do livro B"
    })
    
    # Consultar todos
    response = client.get("/livros")
    assert response.status_code == 200
    
    dados = response.json()
    assert len(dados) == 2  # Deve ter 2 livros

# Teste 3: Buscar livro por título
def test_buscar_por_titulo():
    """Testa a busca de livros pelo título"""
    # Cadastrar um livro específico
    client.post("/livros", json={
        "titulo": "Harry Potter e a Pedra Filosofal",
        "autor": "J.K. Rowling",
        "data_publicacao": "1997-06-26",
        "resumo": "Primeiro livro da série Harry Potter"
    })
    
    # Buscar por "Harry"
    response = client.get("/livros?titulo=Harry")
    assert response.status_code == 200
    
    dados = response.json()
    assert len(dados) == 1
    assert dados[0]["titulo"] == "Harry Potter e a Pedra Filosofal"

# Teste 4: Buscar livro por autor
def test_buscar_por_autor():
    """Testa a busca de livros pelo autor"""
    # Cadastrar livros do mesmo autor
    client.post("/livros", json={
        "titulo": "Dom Casmurro",
        "autor": "Machado de Assis",
        "data_publicacao": "1899-01-01",
        "resumo": "Resumo do Dom Casmurro"
    })
    
    client.post("/livros", json={
        "titulo": "Memórias Póstumas de Brás Cubas",
        "autor": "Machado de Assis",
        "data_publicacao": "1881-01-01",
        "resumo": "Resumo das Memórias Póstumas"
    })
    
    # Buscar por autor "Machado"
    response = client.get("/livros?autor=Machado")
    assert response.status_code == 200
    
    dados = response.json()
    assert len(dados) == 2  # Encontrou 2 livros do Machado
    assert all(livro["autor"] == "Machado de Assis" for livro in dados)

# Teste 5: Busca que não encontra resultados
def test_busca_sem_resultados():
    """Testa busca que não retorna nenhum livro"""
    response = client.get("/livros?titulo=livroinexistente123")
    assert response.status_code == 200
    assert response.json() == []

# Teste 6: Validar dados inválidos no cadastro
def test_cadastro_dados_invalidos():
    """Testa se a API rejeita dados inválidos"""
    # Título muito curto
    response = client.post("/livros", json={
        "titulo": "",  # Título vazio
        "autor": "Autor",
        "data_publicacao": "2024-01-01",
        "resumo": "Resumo válido"
    })
    assert response.status_code == 422  # Erro de validação
    
    # Resumo muito curto
    response = client.post("/livros", json={
        "titulo": "Título Válido",
        "autor": "Autor",
        "data_publicacao": "2024-01-01",
        "resumo": "Curto"  # Menos de 10 caracteres
    })
    assert response.status_code == 422

# Teste 7: Busca combinada (título + autor)
def test_busca_combinada():
    """Testa busca usando título e autor simultaneamente"""
    client.post("/livros", json={
        "titulo": "O Alienista",
        "autor": "Machado de Assis",
        "data_publicacao": "1882-01-01",
        "resumo": "Resumo do Alienista"
    })
    
    client.post("/livros", json={
        "titulo": "O Cortiço",
        "autor": "Aluísio Azevedo",
        "data_publicacao": "1890-01-01",
        "resumo": "Resumo do Cortiço"
    })
    
    # Buscar título "O" e autor "Machado"
    response = client.get("/livros?titulo=O&autor=Machado")
    assert response.status_code == 200
    
    dados = response.json()
    assert len(dados) == 1
    assert dados[0]["titulo"] == "O Alienista"