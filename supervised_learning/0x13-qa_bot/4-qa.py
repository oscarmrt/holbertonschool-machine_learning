#!/usr/bin/env python3
"""program that answers questions from multiple reference texts"""
question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(coprus_path):
    """function that answers questions from multiple reference texts"""
    in_qa = ''
    while (in_qa is not None):
        in_qa = input("Q: ")
        if in_qa.lower() in ['bye', 'exit', 'quit', 'goodbye']:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, in_qa)
        answer = question_answer(in_qa, reference)
        if answer == '':
            print("A: Sorry, I do not understand your question.")
            continue
        print("A: {}".format(answer))
