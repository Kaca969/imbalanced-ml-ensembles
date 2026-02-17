import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbr

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.feature_selection import SelectFromModel


RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_PATH = os.path.join("data", "heart_failure_clinical_records_dataset.csv")
CILJNI_ATRIBUT = "DEATH_EVENT"


def analyze_model_performance(model, X_test, y_test, model_name, plot_roc=True):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"AUC:       {auc:.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            f"Create folder 'data' and place the CSV as 'heart_failure_clinical_records_dataset.csv'."
        )

    dataset = pd.read_csv(DATA_PATH)

    print("Osnovne informacije o skupu podataka:")
    print(dataset.info())

    print("\nNedostajuće vrednosti po kolonama:\n", dataset.isnull().sum())
    print("\nDeskriptivna statistika:\n", dataset.describe())

    # Korelacioni matriks
    korelacioni_matriks = dataset.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sbr.heatmap(korelacioni_matriks, annot=True, cmap="coolwarm")
    plt.title("Korelacioni matriks")
    plt.tight_layout()
    plt.show()

    # Raspodela ciljnog atributa
    plt.figure(figsize=(8, 6))
    sbr.histplot(dataset[CILJNI_ATRIBUT], bins=2, kde=False)
    plt.title("Raspodela ciljnog atributa (DEATH_EVENT)")
    plt.xlabel("DEATH_EVENT")
    plt.ylabel("Frekvencija")
    plt.tight_layout()
    plt.show()

    # Pite grafikon za pol
    polovi_brojac = dataset["sex"].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(
        polovi_brojac,
        labels=polovi_brojac.index.map({1: "Male", 0: "Female"}),
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Raspodela polova")
    plt.tight_layout()
    plt.show()

    # Boxplotovi po polu
    kljucni_parametri = ["age", "ejection_fraction", "serum_creatinine", "smoking", "time"]
    for parametar in kljucni_parametri:
        plt.figure(figsize=(10, 6))
        sbr.boxplot(x="sex", y=parametar, data=dataset)
        plt.title(f"Raspodela {parametar} po polu")
        plt.xlabel("Pol (0 - Female, 1 - Male)")
        plt.ylabel(parametar)
        plt.tight_layout()
        plt.show()

    # Raspodela starosti vs target
    plt.figure(figsize=(12, 8))
    sbr.lineplot(data=dataset, x="age", y=CILJNI_ATRIBUT, ci=None)
    plt.title("Raspodela starosti u odnosu na ciljni atribut (DEATH_EVENT)")
    plt.xlabel("Starost")
    plt.ylabel("DEATH_EVENT")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sbr.histplot(data=dataset, x="age", bins=30, kde=True, hue=CILJNI_ATRIBUT, multiple="stack")
    plt.title("Raspodela starosti u odnosu na DEATH_EVENT")
    plt.xlabel("Starost")
    plt.ylabel("Frekvencija")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    sbr.boxplot(data=dataset, x=CILJNI_ATRIBUT, y="ejection_fraction")
    plt.title("Raspodela ejection_fraction u odnosu na DEATH_EVENT")
    plt.xlabel("DEATH_EVENT")
    plt.ylabel("Ejection Fraction")
    plt.tight_layout()
    plt.show()

    # Priprema podataka
    X = dataset.drop([CILJNI_ATRIBUT], axis=1)
    y = dataset[CILJNI_ATRIBUT]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # Skaliranje
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Manual oversampling (train only)
    izbroj_class_0, izbroj_class_1 = y_train.value_counts()
    df_class_0 = pd.DataFrame(X_train[y_train == 0])
    df_class_1 = pd.DataFrame(X_train[y_train == 1])

    df_class_1_over = df_class_1.sample(izbroj_class_0, replace=True, random_state=RANDOM_STATE)
    X_train_balanced = pd.concat([df_class_0, df_class_1_over], axis=0)
    y_train_balanced = pd.Series([0] * izbroj_class_0 + [1] * izbroj_class_0)

    plt.figure(figsize=(8, 6))
    sbr.histplot(y_train_balanced, bins=2, kde=False)
    plt.title("Raspodela ciljnog atributa nakon manualnog oversamplinga")
    plt.xlabel("DEATH_EVENT")
    plt.ylabel("Frekvencija")
    plt.tight_layout()
    plt.show()

    # Stacking
    estimators = [
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ]
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    )

    param_grid_stacking = {
        "rf__n_estimators": [50, 100],
        "svm__C": [0.1, 1],
        "final_estimator__C": [0.1, 1, 10],
    }

    # IMBALANCED => scoring='f1' (možeš i 'roc_auc')
    grid_search_stacking = GridSearchCV(
        estimator=stacking_clf,
        param_grid=param_grid_stacking,
        cv=5,
        n_jobs=-1,
        scoring="f1",
    )
    grid_search_stacking.fit(X_train_balanced, y_train_balanced)
    best_stacking_clf = grid_search_stacking.best_estimator_
    print("\nNajbolji hiperparametri za Stacking model:", grid_search_stacking.best_params_)

    # Random Forest
    param_grid_rf = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    grid_search_rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid_rf,
        cv=5,
        n_jobs=-1,
        scoring="f1",
    )
    grid_search_rf.fit(X_train_balanced, y_train_balanced)
    best_rf_clf = grid_search_rf.best_estimator_
    print("\nNajbolji hiperparametri za Random Forest model:", grid_search_rf.best_params_)

    # AdaBoost
    param_grid_ada = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}

    grid_search_ada = GridSearchCV(
        estimator=AdaBoostClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid_ada,
        cv=5,
        n_jobs=-1,
        scoring="f1",
    )
    grid_search_ada.fit(X_train_balanced, y_train_balanced)
    best_ada_clf = grid_search_ada.best_estimator_
    print("\nNajbolji hiperparametri za AdaBoost model:", grid_search_ada.best_params_)

    # ROC krive (sve na jednom plotu)
    plt.figure(figsize=(10, 8))
    analyze_model_performance(best_stacking_clf, X_test, y_test, "Stacking Classifier", plot_roc=True)
    analyze_model_performance(best_rf_clf, X_test, y_test, "Random Forest Classifier", plot_roc=True)
    analyze_model_performance(best_ada_clf, X_test, y_test, "AdaBoost Classifier", plot_roc=True)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Feature importance (RF)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train_balanced, y_train_balanced)

    feature_importances = pd.Series(rf.feature_importances_, index=dataset.drop(columns=[CILJNI_ATRIBUT]).columns)
    plt.figure(figsize=(10, 6))
    feature_importances.nlargest(10).plot(kind="barh")
    plt.title("Značaj atributa (Top 10)")
    plt.tight_layout()
    plt.show()

    # SelectFromModel
    selector = SelectFromModel(rf, threshold="median")
    X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
    X_test_selected = selector.transform(X_test)

    # Stacking sa najbitnijim atributima
    stacking_selected = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    )
    stacking_selected.fit(X_train_selected, y_train_balanced)

    plt.figure(figsize=(10, 8))
    analyze_model_performance(best_stacking_clf, X_test, y_test, "Stacking Classifier (svi atributi)", plot_roc=True)
    analyze_model_performance(stacking_selected, X_test_selected, y_test, "Stacking Classifier (najbitniji atributi)", plot_roc=True)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Feature Selection Comparison)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
