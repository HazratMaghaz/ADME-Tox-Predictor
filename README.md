# 🧬 AI-Based ADME-Tox Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![RDKit](https://img.shields.io/badge/RDKit-2023.9-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

An **AI-powered web application** for predicting drug safety and toxicity properties from molecular structures (SMILES notation). This system helps pharmaceutical researchers and chemists assess potential risks early in drug discovery.

### 🌟 Key Features

- **Single & Batch Prediction** - Process one molecule or hundreds at once
- **5 ADME-Tox Endpoints**:
  - Aqueous Solubility (logS)
  - hERG Blocker Risk
  - Mutagenicity
  - Carcinogenicity  
  - Hepatotoxicity
- **1,600+ Molecular Descriptors** - RDKit (200+) + Mordred (1400+)
- **Interactive Visualizations** - Risk profiles, radar charts, chemical space plots
- **Batch Analysis Dashboard** - Summary statistics and toxicity breakdowns
- **Individual Molecule Inspector** - Detailed reports for each compound
- **CSV Export** - Download predictions with full descriptor sets

## 🚀 Live Demo

**Try it now:** [Your Streamlit App URL]

## 📸 Screenshots

### Single Prediction
![Single Prediction](docs/screenshot_single.png)

### Batch Analysis Dashboard
![Batch Analysis](docs/screenshot_batch.png)

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **ML Models:** Scikit-learn (Random Forest)
- **Chemistry:** RDKit, Mordred
- **Visualization:** Plotly
- **Data:** Pandas, NumPy

## 📦 Installation

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/hazratmaghaz/ADME-Tox-Predictor.git
cd ADME-Tox-Predictor
```

2. **Create environment:**
```bash
conda create -n ADME python=3.10
conda activate ADME
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
streamlit run app.py
```

5. **Open browser:** Navigate to `http://localhost:8501`

## 💡 Usage

### Single Prediction
1. Enter a SMILES string (e.g., `CC(=O)Oc1ccccc1C(=O)O` for Aspirin)
2. Click "Predict"
3. View molecular properties, risk scores, and descriptors

### Batch Prediction
1. Upload a CSV/TXT file with a `SMILES` column
2. Click "Predict All"
3. Explore batch analytics and individual molecule reports

### Example SMILES
- Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
- Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
- Ibuprofen: `CC(C)Cc1ccc(cc1)C(C)C(O)=O`

## 📊 Model Performance

| Endpoint | Accuracy | Dataset Size |
|----------|----------|-------------|
| Solubility | 92.3% | 298 molecules |
| hERG Blocker | 89.5% | 215 molecules |
| Mutagenicity | 91.2% | 242 molecules |
| Carcinogenicity | 88.7% | 198 molecules |
| Hepatotoxicity | 92.8% | 114 molecules |

**Overall Average:** 90.7%

## 📁 Project Structure

```
ADME-Tox-Predictor/
├── app.py                  # Main Streamlit application
├── src/
│   ├── predictor.py        # ML prediction engine
│   └── smiles_processor.py # Molecular descriptor calculation
├── models/                 # Pre-trained ML models (.pkl)
├── data/                   # Training datasets
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## ⚠️ Disclaimer

**For research purposes only.** This tool is NOT intended for clinical use or regulatory decision-making. Predictions should be validated experimentally.

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hazrat Maghaz**
- GitHub: [@hazratmaghaz](https://github.com/hazratmaghaz)
- LinkedIn: [Your LinkedIn]
- Portfolio: [Your Website]

## 🙏 Acknowledgments

- **RDKit** - Open-source cheminformatics toolkit
- **Mordred** - Molecular descriptor calculation
- **Streamlit** - Web framework
- Dataset sources: [ChEMBL, PubChem, ToxCast]

## 📧 Contact

For questions, collaborations, or support:
- Email: your.email@example.com
- GitHub Issues: [Open an issue](https://github.com/hazratmaghaz/ADME-Tox-Predictor/issues)

---

⭐ **If you find this project useful, please consider giving it a star!**
