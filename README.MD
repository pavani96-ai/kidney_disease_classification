# Kidney Disease Classification Pipeline (End-to-End Deep Learning) 🔬

## 📝 Project Overview
This project delivers a production-grade, End-to-End Deep Learning and MLOps solution designed to assist medical professionals in classifying kidney conditions from medical imaging. The architecture is built as a multi-class classifier capable of accurately detecting abnormalities such as **Tumors, Stones, and Normal** kidney structures. 

Moving beyond isolated notebook scripts, this repository follows a clean, decoupled package structure, implements full data and pipeline version control, integrates remote experiment tracking, and packages the model into a containerized API ready for automated cloud deployment.

---

## ⚙️ ML Pipeline Workflow
The project follows a highly structured, step-by-step modular architecture:

1. **Data Ingestion**: 
   * Automates the retrieval of medical imaging datasets and manages local artifact storage structures.
   * Performs a structured split to create distinct Training, Validation, and Test sets.
2. **Data Transformation (Preparation)**: 
   * Implements preprocessing layers including pixel normalization, input scaling, and resizing to match model graph expectations.
   * Performs real-time training data augmentation to prevent overfitting.
3. **Model Training (Transfer Learning)**: 
   * Loads a pre-trained, lightweight **MobileNet** backbone and appends custom dense, dropout, and Softmax classification layers.
   * Compiles and fine-tunes the network using optimized hyperparameters managed via centralized configuration files (`config/params.yaml`).
4. **Model Evaluation & Remote Tracking**:
   * Evaluates the fine-tuned model against validation sets, calculating strict performance metrics (Accuracy, Loss, Precision, Recall, and F1-Score).
   * Streams training logs, metrics, hyperparameter structures, and final model weights directly to a remote **MLflow** registry hosted via **DagsHub**.
5. **Deployment**:
   * Wraps the finalized model components inside a high-throughput **FastAPI** web endpoint framework configured to accept file uploads (medical images) and return prediction arrays in a clean tabular format.
   * Deployed systematically onto an **AWS EC2** instance via Docker containers, utilizing a dedicated **AWS IAM User** with fine-grained access policies to securely push artifacts to **AWS S3** and Docker images to **AWS ECR**.

---

## 🛠️ Step-by-Step Development Workflow
To modify, scale, or add new components to this pipeline, follow this strict 10-step engineering workflow:

1. **Update `config/config.yaml`**: Define the base data paths and directory configurations for the stage.
2. **Update `.env` [Optional]**: Configure environment variables, database keys, or remote access tokens securely.
3. **Update `config/params.yaml`**: Centralize architectural parameters and hyperparameter constants (epochs, batch size, learning rate).
4. **Update the Entity**: Define type-safe configuration dataclasses inside `src/kidney_disease_classification/entity`.
5. **Update the Configuration Manager**: Add relevant component configuration methods inside `src/kidney_disease_classification/config/configuration.py`.
6. **Update the Components**: Implement the operational Python code (e.g., Ingestion, Transformation, Trainer) inside `src/kidney_disease_classification/components`.
7. **Update the Pipeline**: Chain components together into execution routines within `src/kidney_disease_classification/pipeline`.
8. **Update `main.py`**: Add execution triggers to expose the step to the global orchestration loop.
9. **Update `dvc.yaml`**: Define the stage inputs, outputs, parameters, and commands to preserve step caching and complete pipeline repeatability.
10. **Update `app.py`**: Expose inference logic or updated payload structures to the production **FastAPI** application endpoints.

---

## 📊 Accomplishments & Experimental Results
- **Champion Architecture**: Built upon a **MobileNet Transfer Learning** framework for lightweight, efficient, and highly accurate edge or cloud inference.
- **Evaluation Metrics**: Achieved a stellar **Validation Accuracy of 97%** and a matching **F1-Score of 0.97**.
- **MLOps Maturity**:
  - Full dataset tracking and tracking variations handled via **DVC (Data Version Control)** to eliminate data sync issues.
  - Pushes final deep learning model `.h5`/`.keras` binaries and image preprocessor artifacts directly to an **AWS S3 Bucket** backup layer.
  - Automates infrastructure container storage using **AWS ECR (Elastic Container Registry)**.
  - Implements fully automated CI/CD code updates through **GitHub Actions Runners** powered by secure IAM-backed repository secrets and `.env` runtime configurations on an **AWS EC2** instance.

---

## 🛠️ Tech Stack
* **Language**: Python 3.10+
* **Deep Learning Frameworks**: TensorFlow, Keras
* **Data & Image Processing**: OpenCV, NumPy, Pandas, Matplotlib
* **MLOps / DevOps**: DVC, MLflow, DagsHub, Docker, GitHub Actions, Git
* **Cloud Infrastructure**: AWS (EC2, ECR, S3, IAM)
* **API Development**: FastAPI, Uvicorn
* **Configuration Management**: YAML (`pyyaml`)

---

## 📂 Repository Structure
```text
.
├── .github/workflows/  # GitHub Actions CI/CD workflows
├── artifacts/          # Images, preprocessed arrays, and weights (Tracked via DVC)
├── config/             # Centralized configurations (config.yaml, params.yaml)
├── research/           # Jupyter notebooks for model prototyping and EDA
├── src/                # Modular source code package root
│   └── kidney_disease_classification/
│       ├── components/ # Pipeline steps (Ingestion, Transformation, Trainer, Evaluation)
│       ├── entity/     # Type-safe configuration and artifact tracking entities
│       ├── pipeline/   # Orchestration loops (Training and Prediction triggers)
│       └── utils/      # Common utilities (logging, exceptions, model saving helpers)
├── app.py              # FastAPI file-upload and tabular prediction API server entry point
├── main.py             # Global execution script to run the full pipeline
├── requirements.txt    # Project Python dependencies
├── setup.py            # Packaging script for local module installation
└── Dockerfile          # Production-grade Docker container setup

🚀 Getting Started
1. Clone the Repository
Bash
git clone [https://github.com/pavani96-ai/kidney_disease_classification.git](https://github.com/pavani96-ai/kidney_disease_classification.git)
cd kidney_disease_classification

2. Install Environment Dependencies
We use Conda to guarantee runtime binary isolation. Run these commands within your terminal:
Bash
# Create the environment path
conda create -p venv python=3.10 -y
# Activate the local virtual environment
conda activate ./venv
# Install dependencies
pip install -r requirements.txt

3. Environment Configuration
* Create a secure configuration file named .env in the project root directory. Fill out the variable footprints exactly using your personal developer credentials:

  Code snippet
  DAGSHUB_TOKEN="your_dagshub_secret_app_token"
  MLFLOW_TRACKING_URI="your_dagshub_mlflow_url"
  MLFLOW_TRACKING_USERNAME="pavani96-ai"
  MLFLOW_TRACKING_PASSWORD="your_dagshub_secret_app_token"
⚠️ Note: Never commit your finished .env file to public branches. It is ignored by Git by default.

4. Fetch Versioned Assets
*  Synchronize your workspace data features and model weights directly from your remote data vault layout:
  Bash
  dvc pull

5. Run the End-to-End Pipeline
* To trigger ingestion, transformation, training, and remote MLflow experiment logging sequentially:

 Bash
 python main.py

6. Start the Local FastAPI Server
 Bash
 uvicorn app:app --host 0.0.0.0 --port 8080
 Open http://localhost:8080/docs in your browser to interactively upload test images and receive classification predictions instantly via the Swagger UI interface.

🐳 Containerization & Cloud Deployment
* To execute your production application wrapper locally under a matching container ecosystem environment:
  Bash
  # Build the Docker Image package
  docker build -t kidney-disease-classifier:latest .
  # Run the Containerized API with environment contexts mapped
  docker run -p 8080:8080 --env-file .env kidney-disease-classifier:latest

7.Continuous Delivery & Security Architecture
* Merging updates directly into your production main branch immediately invokes an automated GitHub Actions runner flow.
* The automated pipeline securely accesses AWS resources using an AWS IAM User key configuration saved securely within GitHub Secrets to perform the following actions:
* Runs code sanity checks and validation tests.
* Builds a production-grade Docker container footprint.
* Publishes the validated application build directly into your AWS ECR instance registry.
* Backs up the training metadata and final generated model binaries to your dedicated AWS S3 Bucket.
* Automatically issues image refresh commands over to your target cloud AWS EC2 instance host to complete the zero-downtime update.



