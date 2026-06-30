# 🫁 Breath Biomarker Analysis for Lung Cancer and Diabetes

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?style=flat-square&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.1+-blue?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=flat-square&logo=pandas)

An end-to-end Machine Learning and Deep Learning pipeline designed to process real-world Electronic Nose (e-Nose) hardware data for the rapid, non-invasive early detection of Lung Cancer and Diabetes Mellitus.

---

## 📖 Project Overview
Systemic metabolic diseases (Diabetes) and malignant pulmonary conditions (Lung Cancer) excrete distinct Volatile Organic Compounds (VOCs) into human breath. While standard diagnostics rely on invasive blood draws or expensive CT scans, e-Nose technology offers a painless, rapid, and cost-effective alternative.

This project tackles the immense data science challenges of transitioning e-Nose hardware from the lab to the clinic. By addressing extreme class imbalances, missing sensor arrays (NaNs), and hardware thermal drift, this pipeline successfully maps chaotic chemical sensor signals to specific disease indicators using advanced **K-Nearest Neighbors (KNN) data augmentation** and an **Extreme Gradient Boosting (XGBoost)** classifier.

## 🔬 Hardware & Dataset
Data was aggregated from 17 separate laboratory experiments where sensors were exposed to varying concentrations (1 to 10 ppm) of Acetone, Ammonia, Ethanol, and their mixtures.

*   **Commercial Sensors:** Figaro TGS 822, TGS 826, TGS 2600, TGS 2602
*   **Custom Nanomaterials:** Pure ZnO, NiO-ZnO (Acetone optimized), Cr-doped ZnO (Ethanol optimized)
*   **Initial Dataset:** ~78,000 rows of raw, time-series electrical resistance data.

## ⚙️ The Data Pipeline
Real-world hardware data is notoriously messy. This project implemented a rigorous data preprocessing and augmentation engine:

1.  **The "Nuclear" Deep Clean:** Standardized typographical errors across laboratory files (e.g., merging `"Mixed 1 ppm"` and `"mixed 1ppm"`), shrinking 58 fragmented labels into core disease classes.
2.  **The "Clean Air" Paradox:** Purged over **43,000+ redundant rows** of idle sensor time, forcing the models to train strictly on active chemical reaction curves.
3.  **Missing Data (NaNs):** Because sensor setups varied across trials, the matrix contained heavy sparsity.
4.  **Data Augmentation (Balancing):** Standard balancing tools like SMOTE crash on missing data. We implemented two bespoke solutions to equalize classes to 2,000 rows each:
    *   *Parametric Distribution Fitting:* Modeled sensor noise to the 5 best-fitting mathematical curves (Beta, Pearson3, Log-Normal, Cauchy, Uniform).
    *   *KNN Interpolation:* Mean-imputation combined with $k=5$ Euclidean nearest-neighbor vectoring to spawn points exactly between existing real sensor readings.

## 🧠 Machine Learning Architectures
Two primary models were tested to evaluate the augmented hardware data:

*   **1D-CNN (Deep Learning):** An automated feature extractor testing 2 vs. 3 convolutional layers, paired with Dense layers of 128/192/256 neurons, and ReLU/LeakyReLU activations.
*   **XGBoost (Ensemble Learning):** An unrestricted, GPU-accelerated forest of 3,000 decision trees (Max Depth: 16) designed to natively handle the sparse NaN matrix and execute highly non-linear thresholding.

## 📊 Key Discoveries & Results

### 1. The Covariance Phenomenon
During augmentation testing, datasets generated via univariate parametric distributions (Uniform, Cauchy, Beta) caused model accuracy to plummet to **~40%**. Generating data column-by-column destroyed the multivariate covariance between the 7 sensors, producing physically impossible combinations. **KNN interpolation** preserved the cross-reactive physical signatures of the sensors, pushing model accuracy to **94.83%**.

### 2. Model Performance
*   **1D-CNN Peak Accuracy:** `83.37%`
*   **XGBoost Peak Accuracy:** `94.83%`

*Conclusion:* While 1D-CNNs successfully converged, they assume a spatial/temporal hierarchy which is arbitrary in a tabular sensor array. Tree-based gradient boosting (XGBoost) proved to be the vastly superior architecture for non-linear thresholding across independent continuous hardware variables.

## 🚀 Future Scope
The serialized XGBoost model and associated scalers/encoders can be deployed onto Edge-computing devices (e.g., Raspberry Pi, Arduino) integrated directly with the e-Nose hardware. This framework provides the software foundation for a portable, low-latency, real-time clinical screening tool.

---

## 💻 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/Breath-Biomarker-Analysis.git
cd Breath-Biomarker-Analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. To Generate the Datasets**

**A. For Beta, Pearson3, Cauchy etc datasets**
```bash
python data-distribution-syn-data-gen.py
```

**B. To Generate the KNN dataset**
```bash
python knn-data-generation.py
```