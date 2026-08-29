# Next Word Predictor (LSTM)

A configurable LSTM-based next-word prediction model, trained on a text
corpus of your choice from Kaggle, served through a Flask web UI where you
type a starting phrase and watch it generate a continuation, word by word.

![CI/CD](https://github.com/YOUR_USERNAME/next-word-lstm/actions/workflows/main.yaml/badge.svg)

## Architecture: Stacked LSTM

The model is a **stacked LSTM** — two LSTM layers, strictly left-to-right
(causal):

```
Input
Embedding
LSTM(150, return_sequences=True)
Dropout
LSTM(100)
Dropout
Dense(vocab_size, softmax)
```

This is the standard architecture for this task: enough capacity to model
longer-range dependencies in language, while remaining causal — which
matters because next-word generation is **autoregressive**. At inference
time you predict one word, append it, and predict the next; you only ever
have *past* context, never future context. A bidirectional LSTM's backward
pass depends on future context that simply doesn't exist during
generation, so it was deliberately left out rather than offered as an
option — it doesn't match how this model is actually used, regardless of
how it might score on offline held-out accuracy.

Layer sizes and dropout are configurable via `ModelParams` in `params.yaml`.

## Project Structure

```
next-word-lstm/
├── .github/workflows/main.yaml       # CI/CD: test → build & push to ECR → deploy
├── config/
│   └── config.yaml                    # WHERE artifacts + dataset live
├── research/
│   └── trials.ipynb                   # Experimentation notebook
├── src/
│   └── nextWordPredictor/
│       ├── components/
│       │   ├── data_ingestion.py       # Kaggle API download OR local file
│       │   ├── data_transformation.py  # n-gram sequence generation
│       │   ├── model_trainer.py         # builds & trains chosen LSTM variant
│       │   └── model_evaluation.py      # loss, accuracy, perplexity
│       ├── config/configuration.py      # reads config.yaml + params.yaml
│       ├── constants/
│       ├── entity/                      # dataclasses for each stage's config
│       ├── logging/
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_transformation.py
│       │   ├── stage_03_model_trainer.py
│       │   ├── stage_04_model_evaluation.py
│       │   └── prediction.py            # autoregressive text generation
│       └── utils/common.py
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── app.py                             # Flask app (UI + /train + /predict)
├── index.html                         # Web UI template
├── main.py                            # Runs all training stages in sequence
├── params.yaml                        # Architecture + hyperparameters
└── requirements.txt
```

## Getting the Dataset (Kaggle)

Pick any "next word prediction" / large text corpus dataset on Kaggle —
examples that work well: book/story text corpora, news headline
collections, or any dataset with a large plain-text or single-text-column
CSV file. There are two ways to get it into this project:

**Option A — Manual download (simplest, no credentials needed)**

1. Download your chosen dataset from Kaggle.
2. Place the file at `artifacts/data_ingestion/corpus.txt` (or update
   `local_data_file` in `config/config.yaml` to match your filename).
3. If it's a `.csv` instead of `.txt`, also set `text_column` in
   `config/config.yaml` to the name of the column containing the text.

**Option B — Automatic download via Kaggle API**

1. Get your Kaggle API token: Kaggle account settings → **Create New Token**
   → downloads `kaggle.json`.
2. Set it as environment variables (don't commit `kaggle.json` to git):
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_key
   ```
3. In `config/config.yaml`, set `kaggle_dataset: "owner/dataset-slug"`
   (found in the dataset's Kaggle URL).
4. Run the pipeline — `data_ingestion.py` will download and unzip it
   automatically into `artifacts/data_ingestion/`.

Either way, once the file exists at the configured path, the rest of the
pipeline (tokenization, training, evaluation) is dataset-agnostic.

## How the Pipeline Flows

```
main.py
  │
  ├─▶ stage_01_data_ingestion     → fetches/verifies the raw corpus file
  ├─▶ stage_02_data_transformation → cleans text, builds n-gram sequences,
  │                                   fits tokenizer, splits train/test
  ├─▶ stage_03_model_trainer       → builds the stacked LSTM, trains
  └─▶ stage_04_model_evaluation    → reports loss, accuracy, perplexity
```

Run any single stage in isolation:

```bash
python -m src.nextWordPredictor.pipeline.stage_02_data_transformation
```

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/next-word-lstm.git
cd next-word-lstm
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set up your dataset (see above), then:

```bash
python main.py        # runs all 4 training stages
python app.py          # http://localhost:5000
```

Type a starting phrase, choose how many words to generate, and click
**Generate**. Toggle **Creative mode** for sampled (more varied) output
instead of greedy (deterministic) predictions.

## Running with Docker

```bash
python main.py   # train first, so artifacts/ exists on the host

docker build -t next-word-lstm .
docker run -p 5000:5000 -v $(pwd)/artifacts:/app/artifacts next-word-lstm
```

## CI/CD

`.github/workflows/main.yaml` runs three jobs on every push to `main`:
Continuous Integration (lint/compile check) → Continuous Delivery (build &
push Docker image to Amazon ECR) → Continuous Deployment (self-hosted
runner pulls and restarts the container).

**Required repository secrets:** `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY_NAME`,
`AWS_ECR_LOGIN_URI`. The deployment job needs a self-hosted runner with
Docker installed.

## Notes on Design Decisions

1. **Pre-padding, always.** Sequences are padded with `padding="pre"` so
   real content ends exactly at the position being predicted — this
   matters for the same reason it mattered in the sentiment-analysis
   sibling project: an LSTM's final hidden state (what actually gets read
   for the prediction) is most influenced by whatever it processed last.

2. **`sparse_categorical_crossentropy`, not one-hot + categorical
   crossentropy.** With a vocabulary of thousands of words, one-hot
   encoding every label would be extremely memory-inefficient. Sparse
   categorical crossentropy takes integer labels directly.

3. **Corpus capped via `max_lines`.** N-gram expansion is roughly quadratic
   in average sentence length per line, so very large corpora can make
   preprocessing memory/time-prohibitive on a laptop or CI runner. Raise
   `max_lines` in `params.yaml` if you have more compute available.

4. **Greedy vs. sampled generation.** `temperature=0.0` (default) always
   picks the single most likely next word — deterministic but can loop or
   repeat. "Creative mode" in the UI samples from the probability
   distribution with `temperature=0.7`, which produces more varied,
   less repetitive continuations at the cost of occasional less-coherent
   choices.

## License

MIT — see [LICENSE](LICENSE).
