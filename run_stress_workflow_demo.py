#!/usr/bin/env python3
"""
Quick Stress Notebook Demo Script
This script demonstrates the stress prediction workflow with visualizations
that have been added to stress.ipynb
"""

# 1. Imports & Config
import os, json, time, datetime as dt, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib

RANDOM_STATE=42; np.random.seed(RANDOM_STATE)
DATA_PATH=Path('data/raw/stress_data.csv')
TARGET='Stress Label'
MODEL_DIR=Path('models'); MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR=Path('reports/results'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print('✓ Config ready.')

# 2. Load & Clean
if not DATA_PATH.exists(): raise FileNotFoundError(DATA_PATH)
df=pd.read_csv(DATA_PATH)
if TARGET not in df.columns: raise KeyError(f'Missing target {TARGET}')
for c in df.select_dtypes(include='object').columns: 
    df[c]=df[c].astype(str).str.strip().replace({'nan':np.nan,'None':np.nan,'' : np.nan})
print(f'✓ Loaded data. Shape: {df.shape}')
print(f'✓ Target classes: {df[TARGET].unique()}')

# 3. Preprocess
def split_num_cat(data,target):
    nums,cats=[],[]
    for col in data.columns:
        if col==target: continue
        (nums if pd.api.types.is_numeric_dtype(data[col]) else cats).append(col)
    return nums,cats

num_cols,cat_cols=split_num_cat(df,TARGET)
from sklearn.impute import SimpleImputer
num_pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
cat_pipe=Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre=ColumnTransformer([('num',num_pipe,num_cols),('cat',cat_pipe,cat_cols)])
label_encoder=LabelEncoder(); y=label_encoder.fit_transform(df[TARGET].astype(str)); X=df.drop(columns=[TARGET])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RANDOM_STATE)
print(f'✓ Features: {len(num_cols)} numeric, {len(cat_cols)} categorical')
print(f'✓ Train/Test split: {X_train.shape} / {X_test.shape}')

# 4. Baseline
def metrics(y_true,y_pred): 
    return {'accuracy':accuracy_score(y_true,y_pred),
            'precision_weighted':precision_score(y_true,y_pred,average='weighted',zero_division=0),
            'recall_weighted':recall_score(y_true,y_pred,average='weighted',zero_division=0),
            'f1_weighted':f1_score(y_true,y_pred,average='weighted',zero_division=0)}

baseline=Pipeline([('pre',pre),('clf',DummyClassifier(strategy='most_frequent'))])
baseline.fit(X_train,y_train); base_pred=baseline.predict(X_test)
baseline_metrics = metrics(y_test, base_pred)
print(f'✓ Baseline accuracy: {baseline_metrics["accuracy"]:.4f}')

# 5. Train Candidates
print('\nTraining models:')
cands={'logreg':LogisticRegression(max_iter=400),
       'rf':RandomForestClassifier(n_estimators=120,random_state=RANDOM_STATE),
       'gb':GradientBoostingClassifier(random_state=RANDOM_STATE),
       'svc':SVC(probability=True,random_state=RANDOM_STATE)}
results={}; models={}
for name,clf in cands.items():
    pipe=Pipeline([('pre',pre),('clf',clf)])
    t0=time.time(); pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
    m=metrics(y_test,pred); m['train_time_sec']=round(time.time()-t0,2)
    results[name]=m; models[name]=pipe
    print(f'  {name}: acc={m["accuracy"]:.4f}, f1={m["f1_weighted"]:.4f}, time={m["train_time_sec"]:.2f}s')

results_df=pd.DataFrame(results).T.sort_values('f1_weighted',ascending=False)

# 6. Best Model & Report
best_name=results_df.index[0]; best_model=models[best_name]
y_best=best_model.predict(X_test)
print(f'\n✓ Best model: {best_name}')

print('\nClassification Report:')
print(classification_report(y_test,y_best,zero_division=0,target_names=label_encoder.classes_))

cm=confusion_matrix(y_test,y_best)
plt.figure(figsize=(4,4)); sns.heatmap(cm,annot=True,fmt='d',cmap='Oranges'); 
plt.title('Stress Confusion Matrix'); plt.tight_layout(); 
plt.savefig(RESULTS_DIR/'stress_confusion_matrix.png'); plt.show()

# 7. Save Artifacts
artifact={'model':best_model,'label_encoder':label_encoder,'classes':list(label_encoder.classes_),
          'metrics_table':results_df.to_dict(),'best_model_name':best_name,
          'timestamp':dt.datetime.utcnow().isoformat()}
joblib.dump(artifact, MODEL_DIR/'best_stress_model_fast.joblib')
results_df.to_csv(RESULTS_DIR/'stress_model_metrics_fast.csv')

sample=best_model.predict(X_test[:5])
print(f'✓ Sample predictions: {label_encoder.inverse_transform(sample)}')
print(f'✓ Saved artifacts: {MODEL_DIR}/best_stress_model_fast.joblib')
print(f'✓ Saved metrics: {RESULTS_DIR}/stress_model_metrics_fast.csv')

print(f'\n{"="*60}')
print("STRESS PREDICTION COMPLETED SUCCESSFULLY")
print(f'{"="*60}')
