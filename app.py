import pickle
import random
import os
from flask import Flask, render_template, request, jsonify
from eeg_simulator import simulate_eeg, MENTAL_STATES
from intent_classifier import predict_state
from query_engine import run_query_pipeline
from search_engine import search_web

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    clf = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        topic = data.get("topic", "").strip()
        if not topic:
            return jsonify({"error": "No topic provided"})
        state = random.choice(list(MENTAL_STATES.keys()))
        t, eeg_data, label = simulate_eeg(state)
        result = run_query_pipeline(eeg_data, clf, le, topic=topic)
        search_results = search_web(result["query"], n_results=3)
        return jsonify({
            "mental_state": result["mental_state"],
            "confidence": result["confidence"],
            "topic": result["topic"],
            "query": result["query"],
            "results": search_results
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("  🧠  NEURALSEARCH — WEB INTERFACE")
    print(f"  Running on port {port}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)
