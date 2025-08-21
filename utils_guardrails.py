from collections import Counter

# abstract class for guardrails
class Guardrail:
    def __init__(self, name, description):
        self.name = name

    # returns boolean
    def is_triggered(self, response_text, response_tokens) -> bool:
        pass

class RepetitionGuardrail(Guardrail):
    def __init__(self, name, description, n_gram=5, repetition_threshold=5):
        super().__init__(name, description)
        self.n_gram = n_gram
        self.repetition_threshold = repetition_threshold

    def is_triggered(self, response_text, response_tokens) -> bool:
        if len(response_tokens) < self.n_gram:
            return False

        n_grams = [tuple(response_tokens[i:i+self.n_gram]) for i in range(len(response_tokens)-self.n_gram+1)]

        n_gram_counts = Counter(n_grams)
        most_commons = n_gram_counts.most_common(1)
        if len(most_commons) == 0:
            return False
        return most_commons[0][1] >= self.repetition_threshold

class MaxLengthGuardrail(Guardrail):
    def __init__(self, name, description, max_length=1000):
        super().__init__(name, description)
        self.max_length = max_length

    def is_triggered(self, response_text, response_tokens) -> bool:
        return len(response_tokens) >= self.max_length