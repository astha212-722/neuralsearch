import arxiv
import pickle
from eeg_simulator import simulate_eeg, MENTAL_STATES
from intent_classifier import predict_state
from query_engine import run_query_pipeline

def search_web(query, n_results=3):
    """Search using ArXiv."""
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
        return [{"title": "Error", "snippet": str(e), "url": "", "source": "ArXiv"}]

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
        print(f"      Source  : {r.get('source', 'ArXiv')}")
        print(f"      Snippet : {r['snippet'][:120]}...")
        print(f"      🔗 {r['url']}")
    print("\n" + "=" * 55)
