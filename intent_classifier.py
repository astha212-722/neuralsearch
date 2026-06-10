import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from eeg_simulator import simulate_eeg, MENTAL_STATES
from signal_processor import process_eeg, normalize_features

def build_dataset(samples_per_state=100):
    print("  Building dataset...")
    X = []
    y = []
    for state in MENTAL_STATES:
        for _ in range(samples_per_state):
            t, eeg_data, label = simulate_eeg(state)
            features = normalize_features(process_eeg(eeg_data))
            X.append(features)
            y.append(state)
        print(f"    [{state}] — {samples_per_state} samples done")
    return np.array(X), np.array(y)

def train_classifier(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("\n  Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\n  Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return clf, le

def save_model(clf, le):
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("  Model saved to model.pkl")

def predict_state(eeg_data, clf, le):
    features = normalize_features(process_eeg(eeg_data))
    prediction = clf.predict([features])[0]
    probabilities = clf.predict_proba([features])[0]
    state = le.inverse_transform([prediction])[0]
    confidence = probabilities.max() * 100
    return state, confidence

if __name__ == "__main__":
    print("=" * 50)
    print("  NeuralSearch - Step 3: Intent Classifier")
    print("=" * 50)

    X, y = build_dataset(samples_per_state=100)
    print(f"\n  Dataset: {X.shape[0]} samples, {X.shape[1]} features each")

    clf, le = train_classifier(X, y)
    save_model(clf, le)

    print("\n  Testing live predictions...")
    for state in MENTAL_STATES:
        t, eeg_data, label = simulate_eeg(state)
        predicted, confidence = predict_state(eeg_data, clf, le)
        match = "✓" if predicted == state else "✗"
        print(f"    {match} Real: {state:<18} Predicted: {predicted:<18} Confidence: {confidence:.1f}%")

    print("\n  Step 3 complete. Ready for Step 4.")
