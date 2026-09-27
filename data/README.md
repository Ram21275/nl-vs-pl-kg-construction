# ADE data

This repository does not redistribute the paper's dataset files. Obtain the CodeKGC subsampled ADE splits from the links cited by Gajo and Barrón-Cedeño, then place these files here:

```text
data/ade/train_triples.json
data/ade/val_triples.json
data/ade/test_triples.json
```

Each file is a JSON list with this structure:

```json
[
  {
    "text": "Prothipendylhydrochloride-induced priapism: case report.",
    "triple_list": [["priapism", "Adverse_effect", "Prothipendylhydrochloride"]]
  }
]
```

Review and follow the dataset's own usage terms. Dataset files are intentionally ignored by Git.
