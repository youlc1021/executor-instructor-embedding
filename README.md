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
## Code Snippet

```
from docarray import Document, DocumentArray

da = DocumentArray([doc])
r = da.post('jinaai+sandbox://lc/InstructorEmbeddingExecutor:latest')

print(r.to_json())
```
