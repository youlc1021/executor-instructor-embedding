from jina import DocumentArray, Executor, requests
from InstructorEmbedding import INSTRUCTOR
from typing import Dict
import torch

class InstructorEmbeddingExecutor(Executor):
    """InstructorEmbeddingExecutor embeds texts into 768-dim vectors using instructor embedding"""
    def __init__(
            self,
            model_name: str = 'hkunlp/instructor-large',
            batch_size: int = 32,
            output_value: str = 'sentence_embedding',
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = False,
            device: str = 'cpu',
            **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.output_value = output_value,
        self.convert_to_numpy = convert_to_numpy,
        self.convert_to_tensor = convert_to_tensor,
        self.normalize_embeddings = normalize_embeddings,
        self.model = INSTRUCTOR(model_name_or_path=model_name, device=device)

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass

    @requests(on='/')
    def encode(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        for doc in docs:
            # doc.text : instruction
            # doc.matches : sentences
            document_batches_generator = DocumentArray(
                filter(
                    lambda d: d.text,
                    doc.matches
                )
            ).batch(batch_size=parameters.get('batch_size', self.batch_size))

            with torch.inference_mode():
                for batch in document_batches_generator:
                    batch.embeddings = self.model.encode(
                        sentences=[[doc.text, j.text] for j in batch],
                        output_value=parameters.get('output_value',self.output_value[0]),
                        convert_to_numpy=parameters.get('convert_to_numpy',self.convert_to_numpy),
                        convert_to_tensor=parameters.get('convert_to_tensor',self.convert_to_tensor),
                        normalize_embeddings=parameters.get('normalize_embeddings',self.normalize_embeddings))