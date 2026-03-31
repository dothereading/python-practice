import random
from collections import Counter

class Mastermind():
    def __init__(self, num_of_guesses:int = 10):
        self.num_of_guesses = num_of_guesses
        self.answer = [random.randint(1, 6) for _ in range(4)]

    def check_answer(self, guess: str):
        guess = [int(g) for g in guess]
    
        black = 0
        for a, g in zip(self.answer, guess):
            if a == g:
                black += 1

        guess_count = Counter(guess)
        answer_count = Counter(self.answer)
        matches = 0
        for key, count in guess_count.items():
            if key in answer_count:
                matches += min(count, answer_count[key])
        white = matches - black

        return (black, white)

m = Mastermind()
m.answer = [1,2,2,1]
print(m.check_answer("1221"))
print(m.check_answer("2221"))
print(m.check_answer("3312"))
