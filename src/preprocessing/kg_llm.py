from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# Choose ONE LLM option below

# ---- Option A: Ollama (local, recommended) ----
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

# ---- Option B: OpenAI (if you want) ----
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize transformer
transformer = LLMGraphTransformer(llm=llm)


def extract_graph_from_text(text: str):
    """
    Returns structured graph data
    """
    doc = Document(page_content=text)

    try:
        graph_docs = transformer.convert_to_graph_documents([doc])

        triplets = []

        for g in graph_docs:
            for rel in g.relationships:
                head = rel.source.id
                tail = rel.target.id
                relation = rel.type

                triplets.append((head, relation, tail))

        return triplets

    except Exception as e:
        print(f"[KG ERROR] {e}")
        return []