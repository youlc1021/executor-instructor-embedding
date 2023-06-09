from jina import DocumentArray, Executor, requests
from InstructorEmbedding import INSTRUCTOR
import torch

class InstructorEmbeddingExecutor(Executor):
    """InstructorEmbeddingExecutor embeds texts into 768-dim vectors using instructor embedding"""
    def __init__(
            self,
            model_name: str = 'hkunlp/instructor-large',
            batch_size: int = 32,
            output_value: str = 'sentence_embedding',
            normalize_embeddings: bool = False,
            device: str = 'cpu',
            **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.output_value = output_value
        self.normalize_embeddings = normalize_embeddings
        self.model = INSTRUCTOR(model_name_or_path=model_name, device=device)
        self.model.eval()

    @requests
    def encode(self, docs: DocumentArray, parameters: dict, **kwargs):
        for mini_batch in docs.batch(batch_size=parameters.get('batch_size', self.batch_size)):
            batch_input = [[d.tags.get('instruction', parameters.get('instruction', '')), d.text] for d in mini_batch if d.text]
            with torch.inference_mode():
                mini_batch.embeddings = self.model.encode(
                    sentences=batch_input,
                    output_value=parameters.get('output_value', self.output_value),
                    normalize_embeddings=parameters.get('normalize_embeddings', self.normalize_embeddings)
                )