class Link:

    def __init__(self, weight):
        self.weight: float = weight

    def __repr__(self):
        return f"Link(w:{self.weight})"
