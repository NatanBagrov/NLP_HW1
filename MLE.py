class MLE():


    def __init__(self, v, tags, histories) -> None:
        super().__init__(len(tags))
        self.d = {}
        for index in range(0,self.size):
            self.d[tags[index]]=index