import arxiv
import pickle
from serpapi import GoogleSearch
from eeg_simulator import simulate_eeg, MENTAL_STATES
from intent_classifier import predict_state
from query_engine import run_query_pipeline

SERPAPI_KEY = "6d601c81417b6dddc3ec1d38fd926d7c7450c82b59497ea5735fac34478ee342"

def search_google(query, n_results=3):
    """Search the entire web using SerpAPI Google Search."""
    try:
        params = {
            "q": query,
            "num": n_results,
            "api_key": SERPAPI_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])
        return [
            {
                "title": r["title"],
                "snippet": r.get("snippet", ""),
                "url": r["link"],
                "source": "Google"
            }
            for r in organic[:n_results]
        ]
    except Exception as e:
        print(f"  Google error: {e}")
        return []

def search_arxiv(query, n_results=3):
    """Fall back to ArXiv if Google fails."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=n_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(client.results(search))
        return [
            {
                "title": r.title,
                "snippet": r.summary[:200],
                "url": r.entry_id,
                "source": "ArXiv"
            }
            for r in results
        ]
    except Exception as e:
        return []

def search_web(query, n_results=3):
    """Try Google first, fall back to ArXiv."""
    print("  Trying Google...")
    results = search_google(query, n_results)
    if results:
        print("  Google returned results!")
        return results
    print("  Falling back to ArXiv...")
    results = search_arxiv(query, n_results)
    if results:
        print("  ArXiv returned results!")
    return results

def display_results(result, search_results):
    print("\n" + "=" * 55)
    print("  🧠  NEURALSEARCH — RESULTS")
    print("=" * 55)
    print(f"  Mental State  : {result['mental_state'].upper()} ({result['confidence']:.1f}%)")
    print(f"  Topic         : {result['topic']}")
    print(f"  Search Query  : \"{result['query']}\"")
    print("=" * 55)
    if not search_results:
        print("  No results found.")
        return
    for i, r in enumerate(search_results, 1):
        print(f"\n  [{i}] {r['title']}")
        print(f"      Source  : {r.get('source', 'Web')}")
        print(f"      Snippet : {r['snippet'][:120]}...")
        print(f"      🔗 {r['url']}")
    print("\n" + "=" * 55)

if __name__ == "__main__":
    print("=" * 55)
    print("  NeuralSearch - Google + ArXiv Search")
    print("=" * 55)

    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    test_cases = [
        ("artificial intelligence", "focused"),
        ("space exploration",       "visual_imagery"),
        ("brain computer interface","searching"),
    ]

    for topic, state in test_cases:
        print(f"\n  Searching for: [{topic}]...")
        t, eeg_data, label = simulate_eeg(state)
        result = run_query_pipeline(eeg_data, clf, le, topic=topic)
        search_results = search_web(result["query"])
        display_results(result, search_results)
        print()

    print("  Search engine test complete!")
