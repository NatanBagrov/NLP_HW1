class History():
    t2=None
    t1=None
    sentence=[]
    idx = None

    def __init__(self, t2, t1, sentence, idx) -> None:
        super().__init__()
        self.t1=t1
        self.t2=t2
        self.sentence=sentence
        self.idx=idx


