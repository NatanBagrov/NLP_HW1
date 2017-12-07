
from MLE import MLE
from History import History
import numpy as np

class Viterbi:
    mle = None
    tags = None
    v = None
    d = None

    def __init__(self, mle: MLE, allTags, v) -> None:
        super().__init__()
        self.mle = mle
        self.tags = allTags
        self.v = v
        self.d = {}
        self.tagsNum = len(self.tags)
        for tag, idx in zip(self.tags, range(0, self.tagsNum)):
            self.d[tag] = idx
        self.d['*'] = self.tagsNum

    def inference(self, sentence):
        pi = np.empty((len(sentence),len(self.d),len(self.d)))
        bp = np.empty((len(sentence),len(self.d),len(self.d)))
        pi[:,:,:] = 0
        bp[:, :, :] = 0
        pi[0,len(self.d)-1, len(self.d)-1] = 1
        bp[0,len(self.d) - 1, len(self.d) - 1] = None



        for k in range(1, len(sentence)+1):
            for t1, t in zip(self.tags[:], self.tags[:]):
                tmpMax = -1
                tmpMaxT = None
                for t2 in self.tags:
                    history = History(t2,t1,sentence,k-1)
                    mleRes = self.mle.p(history,t,self.v)
                    tmpRes = pi[k-1,self.d[t2],t1] * mleRes
                    if tmpRes > tmpMax:
                        tmpMax, tmpMaxT = tmpRes, t2
                bp[k,self.d[t1],self.d[t]] = self.d[tmpMaxT]
                pi[k,self.d[t1],self.d[t]] = tmpMax
            tn = np.argmax(pi[len(sentence)])
            print(tn)


