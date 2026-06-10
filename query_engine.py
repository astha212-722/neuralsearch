import random
import pickle
from eeg_simulator import simulate_eeg, MENTAL_STATES
from signal_processor import process_eeg, normalize_features
from intent_classifier import predict_state

# ── Query templates per mental state ─────────────────────────────────────────
# Each state maps to a type of search intent

QUERY_TEMPLATES = {
    "searching": [
        "how to {topic}",
        "what is {topic}",
        "best {topic} explained",
        "guide to {topic}",
    ],
    "focused": [
        "deep dive into {topic}",
        "advanced {topic} techniques",
        "{topic} research papers",
        "expert guide {topic}",
    ],
    "recall": [
        "history of {topic}",
        "origin of {topic}",
        "who invented {topic}",
        "timeline of {topic}",
    ],
    "visual_imagery": [
        "{topic} images",
        "what does {topic} look like",
        "{topic} diagram",
        "visualize {topic}",
    ],
    "relaxed": [
        "interesting facts about {topic}",
        "fun {topic} ideas",
        "explore {topic}",
        "discover {topic}",
    ],
    "idle": [
        "{topic}",
        "about {topic}",
        "introduction to {topic}",
        "overview of {topic}",
    ],
}

# ── Sample topics the brain might be thinking about ──────────────────────────
SAMPLE_TOPICS = [
    "artificial intelligence",
    "climate change",
    "space exploration",
    "quantum computing",
    "meditation",
    "neural networks",
    "ocean life",
    "ancient history",
    "music theory",
    "brain computer interface",
]


def construct_query(mental_state, topic=None):
    """Turn a mental state into a search query."""
    if topic is None:
        topic = random.choice(SAMPLE_TOPICS)

    templates = QUERY_TEMPLATES.get(mental_state, QUERY_TEMPLATES["idle"])
    template = random.choice(templates)
    query = template.format(topic=topic)
    return query, topic


def run_query_pipeline(eeg_data, clf, le, topic=None):
    """Full pipeline: EEG → state → query."""
    state, confidence = predict_state(eeg_data, clf, le)
    query, topic = construct_query(state, topic)
    return {
        "mental_state": state,
        "confidence": confidence,
        "topic": topic,
        "query": query,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("  NeuralSearch - Step 4: Query Engine")
    print("=" * 50)

    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    print("\n  Simulating brain → query pipeline...\n")

    for state in MENTAL_STATES:
        t, eeg_data, label = simulate_eeg(state)
        result = run_query_pipeline(eeg_data, clf, le)

        print(f"  Brain State  : {result['mental_state']} ({result['confidence']:.1f}% confidence)")
        print(f"  Topic Detected: {result['topic']}")
        print(f"  Search Query  : \"{result['query']}\"")
        print(f"  {'-' * 45}")

    print("\n  Step 4 complete. Ready for Step 5.")
