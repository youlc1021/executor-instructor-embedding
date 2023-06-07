from jina import DocumentArray, Executor, requests
from InstructorEmbedding import INSTRUCTOR

class InstructorEmbeddingExecutor(Executor):
    """InstructorEmbeddingExecutor embeds texts into 768-dim vectors using instructor embedding"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = INSTRUCTOR('hkunlp/instructor-large')
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass

    @requests(on='/')
    def generate_embedding(self, docs: DocumentArray, **kwargs):
        for i in docs:
            for j in i.matches:
                j.embedding = self.model.encode([[i.text, j.text]], convert_to_tensor=True)[0]
        return docs