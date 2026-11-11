# Emotion-Prediction: Deep Learning NLP Engine with BiGRU & FastAPI

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2.9.2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Uvicorn](https://img.shields.io/badge/ASGI-Uvicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://www.uvicorn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

An end-to-end deep learning Natural Language Processing (NLP) system engineered to classify human emotional nuances from raw text in real time. Powered by a **Bidirectional Gated Recurrent Unit (BiGRU)** neural network trained on the benchmark `dair-ai/emotion` dataset, the system achieves **92.1% classification accuracy** across six fine-grained emotional states: *Sadness*, *Joy*, *Love*, *Anger*, *Fear*, and *Surprise*.

The inference engine is served via an asynchronous, production-ready **FastAPI** backend with eager lifespan model preloading and is paired with a responsive, client-side web application (**Moodline**) featuring real-time probability distribution visualizers.

---

## 📌 Executive Summary & Business Context

Automated emotional intelligence is a core driver for modern AI-enabled user experiences and customer intelligence operations. Traditional sentiment analysis (positive/negative/neutral) lacks the emotional depth necessary to understand customer frustration, user panic, or brand affinity.

### Key Industry Applications
* **Customer Support Triage**: Instantly flags high-intensity emotional states like *Anger* and *Fear* to route urgent tickets directly to human supervisors, reducing customer churn.
* **Mental Health & Wellness Monitoring**: Analyzes mood progression in daily journaling apps to deliver proactive, empathetic recommendations.
* **Social Media Intelligence & Brand Listening**: Quantifies audience emotional responses during product launches or public relations campaigns beyond surface-level sentiment.
* **Emotion-Aware Conversational AI**: Dynamically adjusts LLM tone, empathy level, and conversation strategy based on the speaker's emotional state.

---

## 📊 Model Architecture & Benchmarking

Multiple recurrent neural network paradigms were trained and evaluated on identical test splits using balanced class weighting to handle label skew:

| Architecture | Parameters / Cell Type | Context Window | Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **SimpleRNN** | Vanilla Recurrent Cell | Unidirectional | 26.45% | Baseline |
| **LSTM** | Long Short-Term Memory Cell | Unidirectional | 33.95% | Explored |
| **GRU** | Gated Recurrent Unit | Unidirectional | 28.90% | Explored |
| **BiGRU (Ours)** | **Bidirectional Gated Recurrent Unit** | **Bidirectional Context** | **92.10%** | **Production Selected** |

### Why BiGRU?
1. **Bidirectional Temporal Context**: Standard forward RNNs evaluate words strictly based on prior tokens. Bidirectional processing scans sentences forward and backward simultaneously, capturing semantic nuance and modifier interactions (e.g., *"I can't believe how incredible this turned out"*).
2. **Computational Efficiency**: GRU cells combine the forget and input gates into an update gate, achieving parity with or surpassing LSTM representations while reducing parameter count and memory footprint, leading to sub-millisecond inference times on standard CPU architectures.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Client ["Client Interface"]
        UI["Moodline Web UI (static/index.html)"]
        CLI["CLI / API Consumers (cURL, Python)"]
    end

    subgraph Server ["FastAPI ASGI Service"]
        ROUTER["FastAPI Router (main.py)"]
        LIFESPAN["Lifespan Manager (Eager Model Preloading)"]
        SCHEMA["Pydantic Validation (TextInput, PredictionResponse)"]
        PRE["Text Preprocessor (Regex, Normalization)"]
    end

    subgraph Engine ["Deep Learning Inference Pipeline"]
        TOK["Tokenizer (Artifacts/tokenizer.pkl)"]
        PAD["Sequence Padding (maxlen=50, post-pad)"]
        MODEL["BiGRU Keras Model (Artifacts/BiGRU_Model.keras)"]
        SOFTMAX["Softmax Multi-Class Distribution"]
    end

    UI -->|HTTP POST /predict| ROUTER
    CLI -->|HTTP POST /predict| ROUTER
    LIFESPAN -.->|Preload on Boot| MODEL
    LIFESPAN -.->|Preload on Boot| TOK
    ROUTER --> SCHEMA
    SCHEMA --> PRE
    PRE -->|Cleaned String| TOK
    TOK -->|Token Sequence| PAD
    PAD -->|Padded Matrix (1, 50)| MODEL
    MODEL --> SOFTMAX
    SOFTMAX -->|Top Emotion & Probabilities| ROUTER
    ROUTER -->|JSON Response| UI
    ROUTER -->|JSON Response| CLI
```

---

## 📁 Repository Structure

```
Emotion-Prediction/
├── Artifacts/
│   ├── BiGRU_Model.keras         # Serialized BiGRU neural network model
│   └── tokenizer.pkl             # Fitted Keras text tokenizer
├── static/
│   ├── index.html                # Responsive web application UI (Moodline)
│   ├── script.js                 # Asynchronous client-side UI controller
│   └── style.css                 # Modern CSS design system & visualizers
├── .gitignore                    # Production Git ignore patterns
├── final_clean.ipynb             # Research notebook: EDA, training & evaluation
├── main.py                       # FastAPI application & inference server
├── README.md                     # Comprehensive technical documentation
└── requirements.txt              # Frozen production dependency specifications
```

---

## 🚀 Quickstart: Local Terminal Setup & Execution

### Prerequisites
* **Python**: 3.11.x (recommended: Python 3.11.9)
* **Git**: Installed and configured on your system

### 1. Clone the Repository
```bash
git clone https://github.com/sanyam-pro/Emotion-Prediction.git
cd Emotion-Prediction
```

### 2. Create and Activate Virtual Environment
* **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Local Inference Server
Launch the application using Uvicorn:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Once initialized, access:
* **Interactive Web App**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📡 API Reference & Verification

### 1. Health Status
Check if the service and serialized model artifacts are loaded in memory:
```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "Server is running",
  "model_loaded": true
}
```

### 2. Predict Emotion
Classify the emotional nuance of an input string:
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"I am absolutely thrilled and overjoyed with these results!\"}"
```

**Response:**
```json
{
  "text": "I am absolutely thrilled and overjoyed with these results!",
  "predicted_emotion": "joy",
  "confidence": 0.9842,
  "all_probabilites": {
    "sadness": 0.0021,
    "joy": 0.9842,
    "love": 0.0094,
    "anger": 0.0018,
    "fear": 0.0019,
    "surprise": 0.0006
  }
}
```

---

## 🧪 Model Pipeline Specifications

1. **Preprocessing**:
   - Lowercasing transformation.
   - Contraction normalization and apostrophe removal.
   - Punctuation & non-alphanumeric sanitization via regular expressions.
   - Multi-whitespace collapsing and trimming.
2. **Tokenization & Padding**:
   - Out-of-vocabulary (OOV) handling.
   - Post-sequence truncation and padding set to an optimal maximum length of `50` tokens.
3. **Inference Latency Optimization**:
   - Serially loaded into memory during ASGI lifespan startup.
   - Synchronous CPU-optimized matrix operations eliminating cold-start overhead per request.

---

## 👤 Author & Credits

Engineered by **[Sanyam Girdhar](https://github.com/sanyam-pro)** (`@sanyam-pro`)  
* **GitHub**: [@sanyam-pro](https://github.com/sanyam-pro)  
* **Email**: [sanyamgirdhar99@gmail.com](mailto:sanyamgirdhar99@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
