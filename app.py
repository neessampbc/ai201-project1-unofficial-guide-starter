"""Gradio query interface for The Unofficial Guide.

Run with:
    python app.py
Then open http://localhost:7860

Input:  a plain-language question about a CUNY Hunter College professor.
Output: a grounded answer (only from retrieved student reviews) plus the list
        of source documents the answer was drawn from.
"""

from __future__ import annotations

import gradio as gr

from query import DEFAULT_TOP_K, ask

EXAMPLE_QUESTIONS = [
    "What do students say about Professor Rivera's exams in CSCI 135?",
    "Is Professor Chen's MATH 155 a heavy workload?",
    "Do students recommend Professor Alvarez for ENGL 120?",
    "How are exams graded in Professor Okafor's PSYCH 100?",
    "Which professor is the easiest A for an econ requirement?",
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    try:
        result = ask(question, k=DEFAULT_TOP_K)
    except Exception as exc:  # surface config/runtime errors in the UI
        return f"Error: {exc}", ""

    sources = result["sources"]
    sources_text = (
        "\n".join(f"• {s}" for s in sources)
        if sources
        else "(no sources — question outside the document set)"
    )
    return result["answer"], sources_text


with gr.Blocks(title="The Unofficial Guide — Hunter Professor Reviews") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask about CUNY Hunter College professors. Answers come **only** from "
        "student reviews in the document set, with sources cited. "
        "_(Reviews are synthetic, created for an educational project.)_"
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What do students say about Professor Rivera's exams?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
