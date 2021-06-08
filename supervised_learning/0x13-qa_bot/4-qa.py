#!/usr/bin/env python3
"""Create a basic Q/A input loop"""

question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def qa_bot(corpus_path):
    """Create a basic Q/A input loop
    corpus_path: is the path to the corpus of reference documents
    """
    while True:
        """Create a basic Q/A input loop"""
        question = input("Q: ")
        if question.lower().strip() in ['bye', 'exit', 'quit', 'goodbye']:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, question)
        answer = question_answer(question, reference)
        if answer == '':
            print("A: Sorry, I do not understand your question.")
            continue
        print("A: {}".format(answer))
