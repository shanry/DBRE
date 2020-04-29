import pickle
import codecs
import time
from random import shuffle


class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,pos,l_dist,r_dist,entity_pos,sentlens):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos
        self.sentlens = sentlens

    def shuffle(self):
        indx = list(range(len(self.sentences)))
        shuffle(indx)
        self.sentences = [self.sentences[i] for i in indx]
        self.pos = [self.pos[i] for i in indx]
        self.l_dist = [self.l_dist[i] for i in indx]
        self.r_dist = [self.r_dist[i] for i in indx]
        self.entity_pos = [self.entity_pos[i] for i in indx]
        self.sentlens = [self.sentlens[i] for i in indx]




class OneSentBag(object):
    def __init__(self, entity_pair, sentence, label, pos, l_dist, r_dist, entity_pos, sentlen):
        self.entity_pair = entity_pair
        self.sentence = sentence
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos
        self.sentlen = sentlen


def get_ins(snum, index1, index2, pos, sentlen, filter_h=3, max_l=100):
    pad = int(filter_h/2)
    x = [0]*pad
    pf1 = [0]*pad
    pf2 = [0]*pad
    new_sentlen = pad
    if pos[0] == pos[1]:
        if (pos[1] + 1) < len(snum):
            pos[1] = pos[1] + 1
        else:
            pos[0] = pos[0] - 1
    if len(snum) <= max_l:
        new_sentlen += sentlen
        for i, ind in enumerate(snum):
            x.append(ind)
            pf1.append(index1[i] + 1)
            pf2.append(index2[i] + 1)
    else:
        new_sentlen += max_l
        idx = [q for q in range(pos[0], pos[1] + 1)]
        if len(idx) > max_l:
            idx = idx[:max_l]
            for i in idx:
                x.append(snum[i])
                pf1.append(index1[i] + 1)
                pf2.append(index2[i] + 1)
            pos[0] = 0
            pos[1] = len(idx) - 1
        else:
            for i in idx:
                x.append(snum[i])
                pf1.append(index1[i] + 1)
                pf2.append(index2[i] + 1)

            before = pos[0] - 1
            after = pos[1] + 1
            pos[0] = 0
            pos[1] = len(idx) - 1
            numAdded = 0
            while True:
                added = 0
                if before >= 0 and (len(x) + 1) <= (max_l+pad):
                    x.append(snum[before])
                    pf1.append(index1[before] + 1)
                    pf2.append(index2[before] + 1)
                    added = 1
                    numAdded += 1

                if after < len(snum) and (len(x) + 1) <= (max_l+pad):
                    x.append(snum[after])
                    pf1.append(index1[after] + 1)
                    pf2.append(index2[after] + 1)
                    added = 1
                if added == 0:
                    break
                before = before - 1
                after = after + 1

            pos[0] = pos[0] + numAdded
            pos[1] = pos[1] + numAdded
    while len(x) < max_l+2*pad:
        x.append(0)
        pf1.append(0)
        pf2.append(0)

    if pos[0] > max_l-3:
        pos[0] = max_l-3
    if pos[1] > max_l-2:
        pos[1] = max_l-2
    if pos[0] == pos[1]:
        pos = [pos[0] + 1, pos[1] + 2]
    else:
        pos = [pos[0] + 1,pos[1] + 1]
    return [x, pf1, pf2, pos, new_sentlen]


def make_quasi_data(data, word2id, filter_h, max_l, num_classes, group_size):
    allData = [[] for _ in range(num_classes)]
    for j, ins in enumerate(data):
        entities = ins.entity_pair
        entities = [word2id.get(entities[0], 0)+1, word2id.get(entities[1], 0)+1]
        rel = ins.label
        pos = ins.pos
        sentences = ins.sentences
        ldist = ins.l_dist
        rdist = ins.r_dist
        sentlens = ins.sentlens
        newSent = []
        l_dist = []
        r_dist = []
        entitiesPos = ins.entity_pos
        newent = []
        newsentlen = []
        for i, sentence in enumerate(sentences):
            idx,a,b,e,l = get_ins(sentence, ldist[i], rdist[i], entitiesPos[i], sentlens[i], filter_h, max_l)
            newSent.append(idx[:])
            l_dist.append(a[:])
            r_dist.append(b[:])
            newent.append(e[:])
            newsentlen.append(l)
        newIns = DocumentContainer(entity_pair=entities, sentences=newSent, label=rel, pos=pos, l_dist=l_dist,
                                       r_dist=r_dist, entity_pos=newent, sentlens=newsentlen)
        newIns.shuffle()
        allData[newIns.label[0]].append(newIns)

    label2sents = [[] for _ in range(num_classes)]
    for j, data in enumerate(allData):
        if j !=0 :
            for i, idata in enumerate(data):
                for num_sent in range(len(idata.sentences)):
                    entity_pair = idata.entity_pair
                    sentence = idata.sentences[num_sent]
                    label = idata.label
                    pos = idata.pos[num_sent]
                    l_dist = idata.l_dist[num_sent]
                    r_dist = idata.r_dist[num_sent]
                    entity_pos = idata.entity_pos[num_sent]
                    sentlen = idata.sentlens[num_sent]
                    onsentbag = OneSentBag(entity_pair, sentence,
                                           label, pos,
                                           l_dist, r_dist,
                                           entity_pos, sentlen)
                    label2sents[j].append(onsentbag)
        else:
            for i, idata in enumerate(data):
                label2sents[j].append(idata)
    quasi_insts = []
    for i in range(1, num_classes):
        # print("{}:{},{}".format(i, len(allData[i]), len(label2sents[i])))
        for j in range(len(label2sents[i])//group_size + 1):
            entity_pairs = []
            rel = [i]
            pos = []
            sentences = []
            sentlens = []
            l_dist = []
            r_dist = []
            entitiesPos = []
            for k in range(group_size*j, min(group_size*j+group_size, len(label2sents[i]))):
                # inst = OneSentBag(label2sents[i][k])
                inst = label2sents[i][k]
                entity_pairs.append(inst.entity_pair)
                pos.append(inst.pos)
                sentences.append(inst.sentence)
                l_dist.append(inst.l_dist)
                r_dist.append(inst.r_dist)
                entitiesPos.append(inst.entity_pos)
                sentlens.append(inst.sentlen)
            quasi_Inst = DocumentContainer(entity_pair, sentences, rel, pos, l_dist, r_dist, entitiesPos, sentlens)
            quasi_insts.append(quasi_Inst)
    print("len of quasi_insts:{}".format(len(quasi_insts)))
    quasi_insts += label2sents[0]
    print("len of quasi_insts:{}".format(len(quasi_insts)))
    shuffle(quasi_insts)
    print("number of quasi-bags: {}".format(len(quasi_insts)))
    return quasi_insts


def make_data(data, word2id, filter_h, max_l):
    newData = []
    for j, ins in enumerate(data):
        entities = ins.entity_pair
        entities = [word2id.get(entities[0], 0)+1, word2id.get(entities[1], 0)+1]  # first line of word embedding matrix is for UNK
        rel = ins.label
        pos = ins.pos
        sentences = ins.sentences
        ldist = ins.l_dist
        rdist = ins.r_dist
        sentlens = ins.sentlens
        newSent = []
        l_dist = []
        r_dist = []
        entitiesPos = ins.entity_pos
        newent = []
        newsentlen = []
        for i, sentence in enumerate(sentences):
            idx,a,b,e,l = get_ins(sentence, ldist[i], rdist[i], entitiesPos[i], sentlens[i], filter_h, max_l)
            newSent.append(idx[:])
            l_dist.append(a[:])
            r_dist.append(b[:])
            newent.append(e[:])
            newsentlen.append(l)
        newIns = DocumentContainer(entity_pair=entities, sentences=newSent, label=rel, pos=pos, l_dist=l_dist, r_dist=r_dist, entity_pos=newent, sentlens=newsentlen)
        newData += [newIns]
    return newData


def get_word2id(f):
    word2id = {}
    while True:
        line = f.readline().strip()
        if not line:
            break
        words = line.split()
        word = words[0]
        id = int(words[1])
        word2id[word] = id
    return word2id


if __name__ == "__main__":

    print("load test and train raw data...")
    testData = pickle.load(open('test_temp.pkl', 'rb'), encoding='utf-8')
    trainData = pickle.load(open('train_temp.pkl', 'rb'), encoding='utf-8')

    word2id_f = codecs.open('word2id.txt', 'r', 'utf-8')
    word2id = get_word2id(word2id_f)

    sentence_len = 80
    max_filter_len = 3
    num_classes = 53
    group_size = 4
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print('point 0 time: ' + '\t\t' + str(now))

    train_data = \
        make_quasi_data(trainData, word2id, max_filter_len, sentence_len, num_classes, group_size)
    f = open('train_bc.pkl', 'wb')
    pickle.dump(train_data, f, -1)
    f.close()

    # test_data = make_data(testData, word2id, max_filter_len, sentence_len)
    # f = open('test.pkl','wb')
    # pickle.dump(test_data, f, -1)
    # f.close()

    # pretrain_data = make_data(trainData, word2id, max_filter_len, sentence_len)  # bags
    # f = open('pretrain.pkl', 'wb')
    # pickle.dump(pretrain_data, f, -1)
    # f.close()

