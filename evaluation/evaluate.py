"""
Script to evaluate the RAG pipeline against the eval_set.json dataset.
Calculates basic accuracy and confidence metrics.
"""

import json
import os
import sys

# Add parent dir to path so we can import rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import generate_answer

EVAL_FILE = os.path.join(os.path.dirname(__file__), "eval_set.json")

def run_evaluation():
    if not os.path.exists(EVAL_FILE):
        print(f"Eval file not found: {EVAL_FILE}")
        return

    with open(EVAL_FILE, "r") as f:
        eval_data = json.load(f)

    print(f"Running evaluation on {len(eval_data)} questions...\n")
    
    results = []
    correct_sources = 0
    total_confidence = 0.0

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        expected_source = item["source_document"]
        
        print(f"[{i}/{len(eval_data)}] Q: {question}")
        
        try:
            response = generate_answer(question)
            
            # Check if the expected source is in the citations
            sources_used = [c["source"] for c in response["citations"]]
            source_match = expected_source in sources_used
            
            if source_match:
                correct_sources += 1
                
            confidence = response.get("confidence", 0.0)
            total_confidence += confidence
            
            print(f"  Expected Source: {expected_source}")
            print(f"  Actual Sources: {sources_used}")
            print(f"  Source Match: {'✅' if source_match else '❌'}")
            print(f"  Confidence: {confidence:.2f}")
            print()
            
            results.append({
                "question": question,
                "source_match": source_match,
                "confidence": confidence
            })
            
        except Exception as e:
            print(f"  Error evaluating question: {e}\n")

    # Summary
    print("-" * 40)
    print("EVALUATION SUMMARY")
    print("-" * 40)
    print(f"Total Questions: {len(eval_data)}")
    print(f"Source Match Accuracy: {correct_sources}/{len(eval_data)} ({(correct_sources/len(eval_data))*100:.1f}%)")
    print(f"Average Confidence: {(total_confidence/len(eval_data)):.2f}")


if __name__ == "__main__":
    run_evaluation()
