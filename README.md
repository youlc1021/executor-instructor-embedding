# InstructorEmbeddingExecutor

InstructorEmbeddingExecutor embeds texts into 768-dim vectors using instructor embedding

# Usage

## Set instruction in Document.tags or in parameters

### Set instruction in Document.tags：

```
doc = Document(
        text='3D ActionSLAM: wearable person tracking in multi-floor environments',
        tags={'instruction':'Represent the Science title:'}
    )
```

### Set instruction in parameters:

```
parameters = {  
        'instruction':'Represent the Science title:',
        'output_value':'sentence_embedding',
        'normalize_embeddings':True
    }
```
## Build a FLOW

```
from docarray import Document, DocumentArray
from jina import Flow
from executor import InstructorEmbeddingExecutor
flow = (
    Flow()
    .add(uses=InstructorEmbeddingExecutor, timeout_ready=-1)
)
parameters = {  
        'instruction':'Represent the Science title:',
        'output_value':'sentence_embedding',
        'normalize_embeddings':True
    }
with flow:
    doc1 = Document(
        text='3D ActionSLAM: wearable person tracking in multi-floor environments',
        tags={'instruction':'Represent the Science title:'}
    )
    doc2 = Document(
        text='Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear.',
        tags={'instruction':'Represent the Science title:'}
    )
    da = DocumentArray([doc1,doc2])
    docs = flow.post(on='/', inputs=da, parameters=parameters)
```
