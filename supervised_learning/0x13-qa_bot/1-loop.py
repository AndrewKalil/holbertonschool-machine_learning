#!/usr/bin/env python3
""" takes in input from the user with the prompt Q: and prints A: as a response. """

while True:
    question = input("Q: ")

    if question.lower().strip() in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    print("A:")
