import random
import numpy as np
ft = open('mytrain.txt', 'w')
fd = open('mydev.txt', 'w')
f_source = open('source.txt', 'w')
f_target = open('target.txt', 'w')
source_lst = []
target_lst = []
f = None
lines = open('corrected_train.txt', 'r').readlines()
counter = 0
for each in lines:
    each = each.strip()
    lst = each.split(" ")
    s = lst[0]
    counter += 1
    i = np.random.randint(10)
    if counter%10==i:
        f = fd
    else:
        f = ft

    for each in list(s):
        if each not in source_lst:
            source_lst.append(each)
    
    lst = lst[1:]
    tg = random.choice(lst)
    for each in list(tg):
        if each not in target_lst:
            target_lst.append(each)
    f.write(s+'\t'+tg+'\n')

for each in source_lst:
    f_source.write(each+'\n')

for each in target_lst:
    f_target.write(each+'\n')

ftest = open('mytest.txt', 'w')
lines = open('corrected_dev.txt', 'r').readlines()
for each in lines:
    each = each.strip()
    lst = each.split(" ")
    s = lst[0]
    ftest.write(s+'\n')
