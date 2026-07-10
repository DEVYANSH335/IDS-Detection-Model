import pandas as pd
import numpy as np
import json
import pickle
import deap
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
DATA_PATH = "Darknet_model_ready_fixed.csv"

# ==============================
# 1. Load Dataset
# ==============================
def load_data(path):
    df = pd.read_csv(path)
    return df



# ==============================
# 2. Preprocessing
# ==============================
def preprocess_data(df):

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN
    df.dropna(inplace=True)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Split features and target (assuming last column is target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y



# ==============================
# 3. Feature Selection (GA replacement or statistical method)
# ==============================
def feature_selection(X, y, population_size=30, generations=20):

    n_features = X.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_features,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):

        # Ensure at least one feature is selected
        if sum(individual) == 0:
            return (0,)

        selected = [i for i, bit in enumerate(individual) if bit == 1]

        X_subset = X.iloc[:, selected]

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        score = cross_val_score(
            clf,
            X_subset,
            y,
            cv=5,
            scoring="accuracy"
        ).mean()

        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=generations,
        verbose=True,
    )

    best = tools.selBest(population, k=1)[0]

    selected_indices = [i for i, bit in enumerate(best) if bit == 1]

    selected_features = X.columns[selected_indices].tolist()

    X_selected = X.iloc[:, selected_indices]

    print(f"Selected {len(selected_features)} features")

    return X_selected, selected_features


# ==============================
# 4. Train Model
# ==============================
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, scaler


# ==============================
# 5. Save Artifacts
# ==============================
import os

def save_artifacts(model, scaler, selected_features):

    os.makedirs("model", exist_ok=True)

    with open("model/rf_ids_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("model/selected_features.json", "w") as f:
        json.dump(selected_features, f)



# ==============================
# 6. Main Execution
# ==============================
if __name__ == "__main__":
    DATA_PATH = "Darknet_model_ready_fixed.csv"  # change this

    df = load_data(DATA_PATH)

    X, y = preprocess_data(df)

    X_selected, selected_features = feature_selection(
    X,
    y,
    population_size=30,
    generations=20
)

    model, scaler = train_model(X_selected, y)

    save_artifacts(model, scaler, selected_features)

    print("Training complete. Model artifacts saved.")
