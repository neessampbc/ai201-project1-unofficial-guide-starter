"""Evaluation harness for The Unofficial Guide.

Runs the 5 test questions from planning.md end-to-end and prints, for each:
the question, the expected answer, the retrieved chunks with distance scores,
and the system's actual response. Use the printed results to fill in the
Evaluation Report and Failure Case sections of README.md.

    python evaluate.py
"""

from __future__ import annotations

from query import DEFAULT_TOP_K, ask, retrieve

# 5 test questions with ground-truth answers grounded in the synthetic corpus.
# Q5 is an intentional stress case: students nickname Prof. Zhang "Z", and the
# query uses the nickname instead of the indexed surname.
TEST_QUESTIONS = [
    {
        "question": "What do students say about Professor Rivera's exams in CSCI 135?",
        "expected": (
            "Exams come almost entirely from his lecture slides (not the textbook), "
            "attendance/redoing the in-class examples matters most, and the midterm "
            "is curved while the final is not."
        ),
    },
    {
        "question": "Is Professor Chen's MATH 155 a heavy workload?",
        "expected": (
            "Yes — weekly WebAssign homework (30-40 problems) is a heavy but "
            "consistent load; it counts ~25% of the grade and mirrors the exams, "
            "so it doubles as studying. Hard but fair."
        ),
    },
    {
        "question": "Do students recommend Professor Alvarez for ENGL 120?",
        "expected": (
            "Generally yes — discussions are engaging and she improves your writing, "
            "but she is a tough essay grader with a heavy reading/writing load; "
            "revisions and office hours are key."
        ),
    },
    {
        "question": "How are exams graded in Professor Okafor's PSYCH 100?",
        "expected": (
            "Three non-cumulative multiple-choice exams, plus short attendance "
            "quizzes at the start of lectures and one APA-style paper. Predictable "
            "and considered easy if you show up."
        ),
    },
    {
        "question": "What are Professor Z's projects like in CSCI 235?",
        "expected": (
            "'Z' is the student nickname for Professor Wenjie Zhang. CSCI 235 has "
            "four large, time-consuming C++ projects (linked list, hash table, BST) "
            "graded strictly on style/memory leaks, weighted above the exams, with "
            "no late submissions accepted."
        ),
    },
]


def run() -> None:
    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["question"]
        print("=" * 90)
        print(f"Q{i}: {q}")
        print(f"Expected: {item['expected']}")
        print("\nRetrieved chunks:")
        for r in retrieve(q, k=DEFAULT_TOP_K):
            print(f"  [{r.distance:.3f}] {r.source}")
            print(f"        {r.text[:120].replace(chr(10), ' ')}...")
        result = ask(q, k=DEFAULT_TOP_K)
        print("\nSystem answer:")
        print(f"  {result['answer']}")
        print(f"Sources: {result['sources']}")
        print()


if __name__ == "__main__":
    run()
