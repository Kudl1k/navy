class LSystem:

    def __init__(self, iterations=3):
        self.iterations = max(0, int(iterations))

    def generate(self, rules, axiom):
        mapping = self._normalize_rules(rules)
        current = str(axiom)

        for _ in range(self.iterations):
            current = "".join(mapping.get(symbol, symbol) for symbol in current)

        return current

    @staticmethod
    def _normalize_rules(rules):
        if isinstance(rules, dict):
            return {str(k): str(v) for k, v in rules.items()}

        if isinstance(rules, str):
            separator = "->" if "->" in rules else "="
            if separator in rules:
                left, right = rules.split(separator, 1)
                left = left.strip()
                if left:
                    return {left: right.strip()}

        raise TypeError("rules must be a dict or a string like 'F -> F+F-F'")
