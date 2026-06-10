import pickle
import random
import time
import webbrowser
from eeg_simulator import simulate_eeg, MENTAL_STATES
from intent_classifier import predict_state
from query_engine import run_query_pipeline
from search_engine import search_web, display_results

def load_models():
    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return clf, le

def simulate_thinking():
    print("\n  🧠 Scanning brain signals", end="", flush=True)
    for _ in range(5):
        time.sleep(0.4)
        print(".", end="", flush=True)
    print(" done!")

def open_results(search_results):
    if not search_results:
        return
    for r in search_results:
        if r["url"] and r["title"] != "Error":
            print(f"  🌐 Opening → {r['url']}")
            webbrowser.open(r["url"])
            time.sleep(0.5)

def interactive_mode():
    clf, le = load_models()
    print("\n" + "=" * 55)
    print("  🧠  NEURALSEARCH — INTERACTIVE MODE")
    print("=" * 55)
    print("  Type any topic and NeuralSearch will:")
    print("  → Simulate your brain state")
    print("  → Construct the best search query")
    print("  → Find results and open them instantly")
    print("  Type 'quit' to exit.")
    print("=" * 55)

    while True:
        print()
        topic = input("  💭 What are you thinking about? → ").strip()

        if topic.lower() == "quit":
            print("\n  Goodbye! 🧠\n")
            break

        if not topic:
            print("  Please enter a topic!")
            continue

        simulate_thinking()

        state = random.choice(list(MENTAL_STATES.keys()))
        t, eeg_data, label = simulate_eeg(state)
        result = run_query_pipeline(eeg_data, clf, le, topic=topic)
        search_results = search_web(result["query"], n_results=3)
        display_results(result, search_results)
        open_results(search_results)

if __name__ == "__main__":
    print("=" * 55)
    print("  🧠  NEURALSEARCH — FULL SYSTEM v1.0")
    print("  Brain to Search Engine Pipeline")
    print("=" * 55)
    interactive_mode()
