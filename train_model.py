import numpy as np
import pandas as pd
import pickle  


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from imblearn.combine import SMOTETomek

DATA_FILE = "predictive_maintenance.csv"
MODEL_FILE = "model.pkl"
RANDOM_STATE = 40

print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please place the dataset in the same directory.")
    exit()

print("Preprocessing data...")
df.drop(['Product ID', 'UDI', 'Target'], axis=1, inplace=True)

failure_mapping_train = {
    "No Failure": 0, "Heat Dissipation Failure": 1, "Power Failure": 2,
    "Overstrain Failure": 3, "Tool Wear Failure": 4, "Random Failures": 5
}
df["Failure Type"].replace(failure_mapping_train, inplace=True)

type_mapping_train = {"H": 0, "L": 1, "M": 2}
df["Type"].replace(type_mapping_train, inplace=True)


feature_columns = [
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]
X = df[feature_columns]
y = df["Failure Type"]


X["Type"].replace({v: k for k, v in type_mapping_train.items()}, inplace=True) 

X_train, _, y_train, _ = train_test_split(X, y, random_state=RANDOM_STATE, test_size=0.33, stratify=y)

print("Applying SMOTE-Tomek resampling...")
X_train_smote = X_train.copy()
X_train_smote["Type"].replace(type_mapping_train, inplace=True)

smote = SMOTETomek(random_state=RANDOM_STATE)
X_resampled, y_resampled = smote.fit_resample(X_train_smote, y_train)

X_resampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
X_resampled_df["Type"].replace({v: k for k, v in type_mapping_train.items()}, inplace=True) 

print("Defining preprocessing pipeline...")
categorical_cols = X_resampled_df.select_dtypes(include="object").columns.to_list()
numerical_cols = X_resampled_df.select_dtypes(include=np.number).columns.to_list()

to_log = ["Rotational speed [rpm]", "Tool wear [min]"]
to_scale = ["Air temperature [K]", "Process temperature [K]", "Torque [Nm]"]

categorical_pipe = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
numeric_pipe_log = make_pipeline(PowerTransformer())
numeric_pipe_scale = make_pipeline(StandardScaler())

preprocessor = ColumnTransformer(transformers=[
    ("categorical", categorical_pipe, categorical_cols), 
    ("power_transform", numeric_pipe_log, to_log),
    ("standardization", numeric_pipe_scale, to_scale)
    ],
    remainder='passthrough' 
)

print("Training Random Forest model...")
rfc_model = OutputCodeClassifier(
    RandomForestClassifier(random_state=RANDOM_STATE),
    code_size=6, 
    random_state=RANDOM_STATE
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", rfc_model)
])

pipeline.fit(X_resampled_df, y_resampled)
print("Training complete.")

print("Saving background data sample for SHAP...")
background_sample = X_resampled_df.sample(100, random_state=RANDOM_STATE)
background_sample.to_csv('shap_background_data.csv', index=False)
print("Background data sample saved to shap_background_data.csv")


print(f"Saving the trained pipeline to {MODEL_FILE}...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(pipeline, f)
print("Pipeline saved successfully.")

