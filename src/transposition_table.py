



class TranspositionTable:
    def __init__(self, max_size=100000):
        self.table = {}
        self.max_size = max_size


    def store(self, key, value):
        if len(self.table) >= self.max_size:
            self.table.pop(next(iter(self.table)))  # Remove the oldest entry
        
        self.table[key] = value


    def lookup(self, key):
        return self.table.get(key)
