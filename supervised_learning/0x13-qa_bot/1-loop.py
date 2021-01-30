#!/usr/bin/env python3
"""program that takes in input from the user with the prompt Q: and prints
A: as a response"""

in_q = ''
while (in_q is not None):
    in_q = input("Q: ")
    if in_q.lower() in ['bye', 'exit', 'quit', 'goodbye']:
        print("A: Goodbye")
        break
    print("A: ")
