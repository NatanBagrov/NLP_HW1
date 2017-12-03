class History():
    t2 = None
    t1 = None
    sentence = []
    idx = None

    def __init__(self, t2, t1, sentence, idx) -> None:
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.sentence = sentence
        self.idx = idx

    def __str__(self) -> str:
        return str(self.t2) + "," + str(self.t1) + "," + str(self.sentence) + "," + str(self.idx)

    def __hash__(self) -> int:
        return hash(self.t1) + hash(self.t2) + hash(self.idx) + hash(str(self.sentence))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o,self.__class__):
            return False
        return self.t2 == o.t2 and self.t1 == o.t1 and self.idx == o.idx and self.sentence == o.sentence

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)
