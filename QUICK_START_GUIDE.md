# 🧠 Stress Level Prediction Project

A comprehensive machine learning project for predicting stress levels using physiological and behavioral data. This project provides a complete end-to-end pipeline from data exploration to model deployment.

## 🚀 Quick Start Guide

### 1. **Project Status Check**
```bash
# Verify everything is working
python test_setup.py
```

### 2. **Start Jupyter Lab**
```bash
# Launch Jupyter Lab for interactive analysis
jupyter lab
```

### 3. **Follow the Analysis Workflow**
Run notebooks in this exact order:
1. **01_data_exploration.ipynb** - Understand your data
2. **02_data_cleaning.ipynb** - Clean and preprocess
3. **03_feature_selection.ipynb** - Select important features
4. **04_model_development.ipynb** - Train and compare models
5. **05_model_evaluation.ipynb** - Evaluate and analyze results

---

## 📁 Project Structure

```
stress_level_prediction/
├── 📊 data/
│   ├── raw/                     # 👈 PUT YOUR DATASET HERE
│   ├── processed/               # Cleaned data (auto-generated)
│   └── external/                # Reference datasets
├── 📓 notebooks/               # Analysis notebooks (run in order)
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_evaluation.ipynb
├── 🔧 src/                     # Reusable code modules
├── 🤖 models/                  # Saved trained models
├── 📈 reports/                 # Generated reports & visualizations
├── requirements.txt            # Python dependencies
└── test_setup.py              # Setup verification script
```

---

## 🎯 Your Next Steps (In Order)

### Step 1: Add Your Dataset 📊
```bash
# Place your stress level dataset in the data/raw/ folder
# Supported formats: CSV, Excel, JSON
cp your_stress_data.csv data/raw/
```

**Expected Dataset Format:**
- **Target Column**: Stress level (e.g., 'Low', 'Medium', 'High' or 0, 1, 2)
- **Feature Columns**: Physiological/behavioral measurements
- **Examples**: heart_rate, sleep_hours, work_pressure, exercise_frequency, etc.

### Step 2: Data Exploration 🔍
```bash
# Open the first notebook
jupyter lab notebooks/01_data_exploration.ipynb
```

**What You'll Do:**
- ✅ Load and examine your dataset
- ✅ Check data quality and missing values
- ✅ Visualize data distributions
- ✅ Analyze correlations
- ✅ Generate summary statistics

### Step 3: Data Cleaning 🧹
```bash
# Open the second notebook
jupyter lab notebooks/02_data_cleaning.ipynb
```

**What You'll Do:**
- ✅ Handle missing values
- ✅ Remove duplicates and outliers
- ✅ Encode categorical variables
- ✅ Scale numerical features
- ✅ Save cleaned dataset

### Step 4: Feature Selection ⚡
```bash
# Open the third notebook
jupyter lab notebooks/03_feature_selection.ipynb
```

**What You'll Do:**
- ✅ Apply 5 different feature selection methods
- ✅ Compare selection techniques
- ✅ Create consensus feature set
- ✅ Validate feature importance

### Step 5: Model Development 🤖
```bash
# Open the fourth notebook
jupyter lab notebooks/04_model_development.ipynb
```

**What You'll Do:**
- ✅ Train 5 different ML algorithms
- ✅ Perform cross-validation
- ✅ Compare model performances
- ✅ Select best model
- ✅ Save trained models

### Step 6: Model Evaluation 📈
```bash
# Open the fifth notebook
jupyter lab notebooks/05_model_evaluation.ipynb
```

**What You'll Do:**
- ✅ Comprehensive performance analysis
- ✅ Generate confusion matrices
- ✅ Create ROC and PR curves
- ✅ Analyze feature importance
- ✅ Generate final report

---

## 🛠️ Available Machine Learning Models

Your project includes these pre-configured models:

1. **🌳 Random Forest** - Robust ensemble method
2. **🚀 Gradient Boosting** - High-performance boosting
3. **🌲 Decision Tree** - Interpretable tree-based model
4. **📊 Logistic Regression** - Linear probabilistic model
5. **🎯 Support Vector Machine** - Powerful classification algorithm

---

## 📊 Key Features Ready to Use

### 🔍 **Data Analysis**
- Automated missing value detection
- Outlier identification and treatment
- Correlation analysis with heatmaps
- Distribution visualization
- Summary statistics generation

### ⚡ **Feature Engineering**
- **Correlation-based selection** - Remove redundant features
- **Univariate tests** - Statistical significance testing
- **Recursive Feature Elimination** - Iterative feature ranking
- **Random Forest importance** - Tree-based feature ranking
- **Mutual information** - Non-linear relationships

### 📈 **Model Evaluation Metrics**
- **Accuracy** - Overall correctness
- **Precision** - Positive prediction reliability  
- **Recall** - True positive detection rate
- **F1-Score** - Balanced precision and recall
- **Cross-validation** - Robust performance estimation

---

## 🔧 Customization Options

### **Configuration Settings** (`src/utils/config.py`)
```python
# Modify these settings for your needs:
CV_FOLDS = 5                    # Cross-validation folds
TEST_SIZE = 0.2                 # Train/test split ratio
RANDOM_STATE = 42               # Reproducibility seed
CORRELATION_THRESHOLD = 0.8     # Feature correlation limit
MISSING_VALUE_STRATEGY = 'mean' # How to fill missing values
```

### **Adding Your Own Model**
```python
# In src/models/model_trainer.py, add:
from your_library import YourClassifier

def initialize_models(self):
    self.models = {
        'your_model': YourClassifier(your_parameters),
        # ... existing models
    }
```

---

## 🎨 Visualization Gallery

Your project will automatically generate:

- 📊 **Data Distribution Plots** - Understand feature patterns
- 🔥 **Correlation Heatmaps** - Feature relationships
- 📈 **Model Performance Charts** - Compare algorithms
- 🎯 **Confusion Matrices** - Classification accuracy
- 📉 **ROC Curves** - Model discrimination ability
- 📊 **Feature Importance** - What drives predictions
- 📈 **Learning Curves** - Training progress analysis

---

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### ❌ Import Errors
```bash
# Fix: Reinstall packages
pip install -r requirements.txt
python test_setup.py
```

#### ❌ Dataset Not Found
```bash
# Fix: Check file location
ls data/raw/
# Your dataset should be here
```

#### ❌ Jupyter Not Starting
```bash
# Fix: Restart Jupyter
jupyter lab --port=8888
```

#### ❌ Memory Issues
```python
# Fix: Use data sampling in notebooks
df_sample = df.sample(n=10000, random_state=42)
```

---

## 📚 Learning Resources

### **Understanding Your Results**

- **High Accuracy (>90%)**: Excellent model performance
- **Medium Accuracy (70-90%)**: Good model, consider feature engineering
- **Low Accuracy (<70%)**: Need more data or different approach

### **Feature Importance Interpretation**
- **High importance**: Critical for stress prediction
- **Medium importance**: Useful but not essential
- **Low importance**: Consider removing to simplify model

### **Model Selection Guide**
- **Random Forest**: Best for beginners, robust performance
- **Gradient Boosting**: Highest accuracy potential
- **Logistic Regression**: Most interpretable results
- **SVM**: Good for small datasets
- **Decision Tree**: Easiest to explain to stakeholders

---

## 🏆 Success Criteria

Your project is successful when you achieve:

- ✅ **Data loaded and cleaned** (Notebook 02 complete)
- ✅ **Features selected** (Notebook 03 complete)  
- ✅ **Model trained** (Notebook 04 complete)
- ✅ **Accuracy > 70%** (Notebook 05 results)
- ✅ **Clear feature importance** (Notebook 05 charts)
- ✅ **Saved final model** (models/ directory)

---

## 🚀 Advanced Features (Optional)

Once you complete the basic workflow, explore:

### **Hyperparameter Tuning**
```python
# In notebook 04, uncomment advanced tuning sections
from sklearn.model_selection import GridSearchCV
# Detailed parameter optimization
```

### **Feature Engineering**
```python
# Create new features in notebook 03
df['stress_score'] = df['work_hours'] * df['pressure_level']
df['recovery_ratio'] = df['sleep_hours'] / df['work_hours']
```

### **Ensemble Methods**
```python
# Combine multiple models for better performance
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([('rf', rf_model), ('gb', gb_model)])
```

---

## 📞 Getting Help

### **If You Get Stuck:**

1. **Check the test script**: `python test_setup.py`
2. **Read notebook comments**: Each cell has detailed explanations
3. **Review error messages**: Copy exact error text for debugging
4. **Check data format**: Ensure your dataset matches expected structure

### **Quick Diagnostics:**
```bash
# Check Python environment
python --version

# Check installed packages
pip list

# Check project structure
ls -la

# Verify data location
ls data/raw/
```

---

## 🎉 Final Notes

**Congratulations!** You have a professional-grade machine learning project setup. This framework follows industry best practices and can be adapted for various classification problems beyond stress prediction.

### **After Completion:**
- 📊 Share your results and visualizations
- 🤖 Deploy your best model for real-time predictions  
- 📈 Extend to other health prediction tasks
- 📝 Document your findings and insights

### **Project Benefits:**
- 🔬 **Scientific rigor** - Proper ML methodology
- 📊 **Comprehensive analysis** - Complete pipeline
- 🎯 **Production ready** - Deployment-ready code
- 📈 **Scalable** - Easy to extend and modify
- 🧪 **Reproducible** - Consistent results

---

**Happy Machine Learning! 🤖✨**

*Start with Step 1: Add your dataset to `data/raw/` and begin your ML journey!*
