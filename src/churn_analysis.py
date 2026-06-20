from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco_customer_churn.xlsx"
RESULTS_PATH = ROOT / "data" / "churn_results.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.txt"
FEATURE_IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"
CONFUSION_MATRIX_PATH = ROOT / "assets" / "confusion_matrix.png"
FEATURE_IMPORTANCE_IMAGE_PATH = ROOT / "assets" / "feature_importance.png"


DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
]

EXPORT_COLUMNS = [
    "CustomerID",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Contract",
    "Internet Service",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the IBM Telco churn dataset."""
    return pd.read_excel(path)


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and encode categorical features for model training."""
    prepared = df.copy()
    prepared["Total Charges"] = pd.to_numeric(prepared["Total Charges"], errors="coerce")
    prepared = prepared.dropna(subset=["Total Charges"]).copy()
    prepared["Churn"] = prepared["Churn Label"].map({"Yes": 1, "No": 0})

    model_df = prepared.drop(columns=DROP_COLUMNS, errors="ignore")

    encoders = {}
    for column in model_df.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        model_df[column] = encoder.fit_transform(model_df[column])
        encoders[column] = encoder

    return prepared, model_df


def train_model(model_df: pd.DataFrame):
    X = model_df.drop("Churn", axis=1)
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return model, X, y_test, y_pred, accuracy, report, matrix


def export_power_bi_results(df: pd.DataFrame, model_df: pd.DataFrame, model, X: pd.DataFrame) -> pd.DataFrame:
    output_df = df.loc[model_df.index, EXPORT_COLUMNS].copy()
    output_df["Churn_Predicted"] = model.predict(X)
    output_df["Churn_Probability"] = model.predict_proba(X)[:, 1]
    output_df.to_csv(RESULTS_PATH, index=False)
    return output_df


def save_reports(model, X: pd.DataFrame, accuracy: float, report: str, matrix) -> pd.DataFrame:
    FEATURES_DIRS = [RESULTS_PATH.parent, METRICS_PATH.parent, CONFUSION_MATRIX_PATH.parent]
    for directory in FEATURES_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    feature_importance = (
        pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    METRICS_PATH.write_text(
        f"Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}",
        encoding="utf-8",
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=feature_importance.head(10),
        x="Importance",
        y="Feature",
        color="#4C9BD5",
    )
    plt.title("Top 10 Important Features for Churn")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_IMAGE_PATH, dpi=150)
    plt.close()

    return feature_importance


def main() -> None:
    df = load_data()
    prepared_df, model_df = prepare_model_data(df)
    model, X, y_test, y_pred, accuracy, report, matrix = train_model(model_df)
    output_df = export_power_bi_results(prepared_df, model_df, model, X)
    feature_importance = save_reports(model, X, accuracy, report, matrix)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Power BI export: {RESULTS_PATH} ({output_df.shape[0]} rows)")
    print("Top 5 churn drivers:")
    print(feature_importance.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
