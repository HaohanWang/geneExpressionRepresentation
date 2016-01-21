import numpy as np

# ids = [line.strip() for line in open('../data/ids.txt')]
#
# text = [line.strip() for line in open('../data/ppi_ids.txt')]
#
# ppi = {}
#
# for line in text:
#     items = line.split()
#     id1 = items[0]
#     id2 = items[1]
#     ppi[(id1, id2)] = 0
#
# count = 0
#
# p = []
# n = []
# np.random.seed(1)
#
# for id1 in ids:
#     count += 1
#     if count %100==0:
#         print id1
#     for id2 in ids:
#         if (id1, id2) in ppi:
#             p.append((id1, id2))
#         else:
#             if np.random.random() < 0.00017:
#                 n.append((id1, id2))
#
# print len(n)
# p = p[:12500]
# n = n[:12500]
#
# for i in range(5):
#     f1 = open('../data/split/ids_train_'+str(i+1)+'.txt', 'w')
#     f2 = open('../data/split/ids_test_'+str(i+1)+'.txt', 'w')
#     f1_ = open('../data/split/labels_train_'+str(i+1)+'.txt', 'w')
#     f2_ = open('../data/split/labels_test_'+str(i+1)+'.txt', 'w')
#
#     for k in range(i*2500, (i+1)*2500):
#         f2.writelines(p[k][0]+'\t'+p[k][1]+'\n')
#         f2_.writelines('1\n')
#     for k in range(i*2500, (i+1)*2500):
#         f2.writelines(n[k][0]+'\t'+n[k][1]+'\n')
#         f2_.writelines('0\n')
#
#     for k in range(0, i*2500):
#         f1.writelines(p[k][0]+'\t'+p[k][1]+'\n')
#         f1_.writelines('1\n')
#     for k in range((i+1)*2500, 12500):
#         f1.writelines(p[k][0]+'\t'+p[k][1]+'\n')
#         f1_.writelines('1\n')
#
#     for k in range(0, i*2500):
#         f1.writelines(n[k][0]+'\t'+n[k][1]+'\n')
#         f1_.writelines('0\n')
#     for k in range((i+1)*2500, 12500):
#         f1.writelines(n[k][0]+'\t'+n[k][1]+'\n')
#         f1_.writelines('0\n')
#
#     f1.close()
#     f2.close()
#     f1_.close()
#     f2_.close()

ids = [line.strip() for line in open('../data/ids.txt')]

data = np.loadtxt('../data/ge.csv', delimiter=',')

ge = {}

for i in range(len(ids)):
    ge[ids[i]] = data[i,:]

for i in range(5):
    t1l = []
    t1r = []
    t2l = []
    t2r = []

    #train
    text = [line.strip() for line in open('../data/split/ids_train_'+str(i+1)+'.txt')]
    for line in text:
        items = line.split()
        id1 = items[0]
        id2 = items[1]
        t1l.append(ge[id1])
        t1r.append(ge[id2])
    np.savetxt('../data/split/data_train_'+str(i+1)+'_a.txt', t1l, delimiter=',')
    np.savetxt('../data/split/data_train_'+str(i+1)+'_b.txt', t1r, delimiter=',')

    #test
    text = [line.strip() for line in open('../data/split/ids_test_'+str(i+1)+'.txt')]
    for line in text:
        items = line.split()
        id1 = items[0]
        id2 = items[1]
        t2l.append(ge[id1])
        t2r.append(ge[id2])
    np.savetxt('../data/split/data_test_'+str(i+1)+'_a.txt', t2l, delimiter=',')
    np.savetxt('../data/split/data_test_'+str(i+1)+'_b.txt', t2r, delimiter=',')