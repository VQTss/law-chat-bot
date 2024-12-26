from langchain.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()

def generate_embedding(text: str):
    return embedding_model.embed_query(text)
