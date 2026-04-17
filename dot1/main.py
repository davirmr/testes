# main.py
# Importando os módulos necessários
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import sqlite3
from contextlib import contextmanager
from contextlib import asynccontextmanager

# Modelo de dados para um Livro (usado para validação)
class Livro(BaseModel):
    id: Optional[int] = None
    titulo: str = Field(..., min_length=1, max_length=200, description="Título do livro")
    autor: str = Field(..., min_length=1, max_length=100, description="Nome do autor")
    data_publicacao: date = Field(..., description="Data de publicação (AAAA-MM-DD)")
    resumo: str = Field(..., min_length=10, max_length=1000, description="Resumo do livro")

# Configuração do banco de dados
DATABASE_URL = "biblioteca.db"

# Função para adaptar date para SQLite (corrige o warning do Python 3.12+)
def adapt_date(date_obj):
    return date_obj.isoformat()

# Função para converter SQLite de volta para date
def convert_date(date_bytes):
    return date.fromisoformat(date_bytes.decode())

# Registrar os adaptadores para o tipo date
sqlite3.register_adapter(date, adapt_date)
sqlite3.register_converter("date", convert_date)

# Gerenciador de contexto para conexões com o banco
@contextmanager
def get_db():
    # Usar detect_types para tratar dates corretamente
    conn = sqlite3.connect(DATABASE_URL, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Função para criar a tabela
def init_database():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS livros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titulo TEXT NOT NULL,
                autor TEXT NOT NULL,
                data_publicacao DATE NOT NULL,
                resumo TEXT NOT NULL
            )
        """)
        conn.commit()

# NOVO: Usando lifespan em vez de on_event (FastAPI moderno)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: executado quando a API inicia
    init_database()
    print("✅ Banco de dados inicializado com sucesso!")
    yield
    # Shutdown: executado quando a API encerra (opcional)
    print("🔒 API encerrada")

# Criando a aplicação FastAPI com lifespan
app = FastAPI(
    title="Biblioteca Virtual API",
    description="API para gerenciar livros em uma biblioteca",
    version="2.0.0",
    lifespan=lifespan  # <-- Nova forma de fazer startup/shutdown
)

# Endpoint de teste
@app.get("/")
def root():
    return {"mensagem": "📚 API Biblioteca Virtual - Funcionando!", "status": "online"}

# Endpoint 1: Cadastrar um novo livro
@app.post("/livros", response_model=Livro, status_code=201)
def cadastrar_livro(livro: Livro):
    """Cadastra um novo livro na biblioteca"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO livros (titulo, autor, data_publicacao, resumo)
            VALUES (?, ?, ?, ?)
        """, (livro.titulo, livro.autor, livro.data_publicacao, livro.resumo))
        
        conn.commit()
        livro_id = cursor.lastrowid
        
        cursor.execute("SELECT * FROM livros WHERE id = ?", (livro_id,))
        livro_criado = cursor.fetchone()
        
        return {
            "id": livro_criado["id"],
            "titulo": livro_criado["titulo"],
            "autor": livro_criado["autor"],
            "data_publicacao": livro_criado["data_publicacao"],
            "resumo": livro_criado["resumo"]
        }

# Endpoint 2: Consultar livros
@app.get("/livros", response_model=List[Livro])
def consultar_livros(
    titulo: Optional[str] = Query(None, description="Filtrar por título (busca parcial)"),
    autor: Optional[str] = Query(None, description="Filtrar por autor (busca parcial)")
):
    """Consulta livros cadastrados na biblioteca"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM livros WHERE 1=1"
        params = []
        
        if titulo:
            query += " AND titulo LIKE ?"
            params.append(f"%{titulo}%")
        
        if autor:
            query += " AND autor LIKE ?"
            params.append(f"%{autor}%")
        
        query += " ORDER BY id DESC"
        
        cursor.execute(query, params)
        livros = cursor.fetchall()
        
        return [
            {
                "id": livro["id"],
                "titulo": livro["titulo"],
                "autor": livro["autor"],
                "data_publicacao": livro["data_publicacao"],
                "resumo": livro["resumo"]
            }
            for livro in livros
        ]