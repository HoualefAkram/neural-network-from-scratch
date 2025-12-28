class Link:

    def __init__(self, weight: float, source):
        self.weight: float = weight
        self.source = source

    def __repr__(self):
        return f"Link(w:{self.weight})"
