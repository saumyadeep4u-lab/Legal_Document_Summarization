# ⚖️ Legal Document Summarization Pipeline

**Developer:** Saumyadeep Bardhan 
**Project Type:** NLP / Large Language Model (LLM) Fine-Tuning & Deployment

## 📖 Overview
The Legal Document Summarization project is an enterprise-grade AI pipeline designed to ingest dense, complex legal contracts and instantly generate highly structured, laser-focused executive briefs. 

Built to handle the unique formatting quirks of legal texts (e.g., ALL CAPS titles, clunky section headers, and massive run-on clauses), this application enforces a strict 5-bullet output prioritizing specific operational clauses (like Audit Rights and SLAs) over generic legal boilerplate. Every processed document is automatically logged and compiled into a downloadable HTML Master Report.

## ✨ Core Features
* **Smart Auto-Detection:** Automatically classifies the contract's primary legal focus (e.g., `AUDIT RIGHTS`, `CONFIDENTIALITY / NDA`, `EMPLOYMENT / NON-COMPETE`, `SLA / UPTIME`, `CAP ON LIABILITY`).
* **Intelligent Text Sanitization:** Bypasses common OCR/formatting bugs by successfully splitting preamble text from uncapped titles and filtering out useless section headers (< 8 words).
* **Uncapped Clause Preservation:** Dynamically captures massive 50+ word restrictive covenants and liability clauses without chopping off critical legal caveats.
* **Master Report Generation:** Compiles all processed session data into a beautifully formatted, colorful, and exportable `.HTML` master report.

## 🗂️ Project Structure
This repository is organized to separate model fine-tuning, source code, testing, and deployment:

```text
legal_summarization_project/
│
├── app.py                      # Main Streamlit web application
├── main.py                     # Entry point for backend execution
├── config.yaml                 # Configuration parameters for models and paths
├── requirements.txt            # Python dependencies
│
├── app_findings/               # Directory for generated HTML Master Reports
│
├── src/                        # Core application logic
│   ├── data_loader.py          # Data ingestion scripts
│   ├── preprocessor.py         # Text sanitization and chunking
│   ├── summarizer.py           # LLM interaction and bullet-point enforcement
│   ├── fine_tuner.py           # Scripts for fine-tuning the base model
│   ├── evaluator.py            # Model performance evaluation metrics
│   ├── generate_pdf.py         # PDF generation utility
│   └── validate_10_contracts.py# Batch validation script
│
├── notebooks/                  # Jupyter notebooks for research and EDA
│   ├── 01_eda_and_chunking_justification.ipynb
│   └── 02_legal_model_fine_tuning.ipynb        
│
├── tests/                      # Unit tests
│   ├── __init__.py            
│   └── test_preprocessor.py    # Ensures sanitization logic remains intact
│
├── deliverables/               # Final reports and figures
│   ├── 5_Contract_Summaries.pdf
│   └── figures/                
│
├── data/                       # Datasets directory (Hosted Externally)
│   ├── raw/                    # Original, untouched data
│   │   ├── CUAD_v1/            
│   │   ├── CUAD_v1.zip         
│   │   └── CUADv1.json         
│   └── processed/              # Cleaned and processed data outputs
│       └── 10_contract_validation_results.csv
│
└── models/                     # Contains fine-tuned model checkpoints (Hosted Externally)
   └── checkpoints/
```

## ☁️ Hosted Resources (Large Files)
To maintain a lightweight repository, large model checkpoints and datasets have been hosted externally via Google Drive. 

* **Raw Datasets (`/data/raw`):** [https://drive.google.com/drive/folders/15vMht6KGFCg2cHl-KihhvwlsxqU0uO4N?usp=drive_link]
* **Fine-Tuned Models (`/models`):** [https://drive.google.com/drive/folders/1XY87i_TIbZ0oJDl1Q8v4CExYBPmyAzQU?usp=drive_link]

*(Note: Download these files and place them in their respective root directories before running the fine-tuning or evaluation scripts).*

## 🚀 Installation & Usage

**1. Create a virtual environment:**
```bash
python -m venv venv
```

**2. Activate the virtual environment:**
```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit Web App:**
```bash
streamlit run app.py
```

*Once running, navigate to `http://localhost:8501` in the browser. Paste any complex legal contract into the text area, and the engine will instantly extract the primary legal focus and generate a compliant 5-point executive brief.*

**5. Run Backend Execution (CLI):**
   If preferred to run the pipeline programmatically without the web interface, execute the main script:

   ```bash
   python main.py
   ```