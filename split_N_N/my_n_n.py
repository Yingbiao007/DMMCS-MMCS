import os

lef = {}  # 存储{(h,r):[t1,t2,t3...]}
rig = {}  # 存储{(r,t):[t1,t2,t3...]}
rellef = {}  # 存储{  关系id{ 左头1：1， 左头2：1}  }
relrig = {}  # 存储{  关系id{ 右尾1：1， 右尾2：1}  }

root = 'E:\\thesis_for_me\\thesis_codes\\DMMCS\\data\\YAGO3-10'
train_path = os.path.join(root, 'train.txt')
valid_path = os.path.join(root, 'valid.txt')
test_path = os.path.join(root, 'test.txt')

triple = open(train_path, "r")
valid = open(valid_path, "r")
test = open(test_path, "r")

with open(train_path, "r", encoding='utf-8') as triple_in:
    for line in triple_in:
        line = line.strip().split('\t')
        if len(line) != 3:
            continue
        h, r, t = line
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

with open(test_path, "r", encoding='utf-8') as test_in:
    for line in test_in:

        line = line.strip().split('\t')
        if len(line) != 3:
            continue
        h, r, t = line
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

with open(valid_path, "r", encoding='utf-8') as valid_in:
    for line in valid_in:
        line = line.strip().split('\t')
        if len(line) != 3:
            continue
        h, r, t = line
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)].append(t)
        rig[(r, t)].append(h)
        if not r in rellef:
            rellef[r] = {}
        if not r in relrig:
            relrig[r] = {}
        rellef[r][h] = 1
        relrig[r][t] = 1

f = open("type_constrain.txt", "w", encoding='utf-8')
f.write("%d\n" % (len(rellef)))

for i in rellef:
    # 关系id、含有此关系及头三元组个数
    f.write("%s\t%d" % (i, len(rellef[i])))
    # 含有此关系的所有头
    for j in rellef[i]:
        f.write("\t%s" % (j))
    f.write("\n")
    # 含有此关系及尾三元组个数
    f.write("%s\t%d" % (i, len(relrig[i])))
    # 含有此关系的所有尾
    for j in relrig[i]:
        f.write("\t%s" % (j))
    f.write("\n")
f.close()

rellef = {}
totlef = {}
relrig = {}
totrig = {}
# lef = {}  # 存储{(h,r):[t1,t2,t3...]}
# rig = {}  # 存储{(r,t):[t1,t2,t3...]}
for i in lef:
    # i[1]是关系，关系不在左边的字典中
    if not i[1] in rellef:
        rellef[i[1]] = 0
        totlef[i[1]] = 0
    # relleft {r1:len(h and r in left)}
    rellef[i[1]] += len(lef[i])  # 有多少个三元组含有r
    # totleft {r1:len(length of differrnt head included r)}
    totlef[i[1]] += 1.0  # 有多少个头含有r

for i in rig:
    if not i[0] in relrig:
        relrig[i[0]] = 0
        totrig[i[0]] = 0
    relrig[i[0]] += len(rig[i])
    totrig[i[0]] += 1.0

s11 = 0
s1n = 0
sn1 = 0
snn = 0
num_of_test = 0
with open(test_path, 'r', encoding='utf-8') as test_in:
    for line in test_in:
        num_of_test += 1
        line = line.strip().split('\t')
        if len(line) != 3:
            continue
        h, r, t = line

        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            s11 += 1
        if (rign >= 1.5 and lefn < 1.5):
            s1n += 1
        if (rign < 1.5 and lefn >= 1.5):
            sn1 += 1
        if (rign >= 1.5 and lefn >= 1.5):
            snn += 1

root = 'D:\\python_project\\test_data\\YAGO3-10'
rootall = os.path.join(root, "fb15kall.txt")
root11 = os.path.join(root, "1-1.txt")
root1n = os.path.join(root, "1-n.txt")
rootn1 = os.path.join(root, "n-1.txt")
rootnn = os.path.join(root, "n-n.txt")
f11 = open(root11, "w", encoding='utf-8')
f1n = open(root1n, "w", encoding='utf-8')
fn1 = open(rootn1, "w", encoding='utf-8')
fnn = open(rootnn, "w", encoding='utf-8')
fall = open(rootall, "w", encoding='utf-8')

fall.write("%d\n" % (num_of_test))
f11.write("%d\n" % (s11))
f1n.write("%d\n" % (s1n))
fn1.write("%d\n" % (sn1))
fnn.write("%d\n" % (snn))
with open(test_path,'r', encoding='utf-8') as test_in:
    for content in test_in:
        
        line = content.strip().split()
        if len(line) != 3:
            continue
        h, r, t = line
        rign = rellef[r] / totlef[r]
        lefn = relrig[r] / totrig[r]
        if (rign < 1.5 and lefn < 1.5):
            f11.write(content)
            fall.write("0" + "\t" + content)
        if (rign >= 1.5 and lefn < 1.5):
            f1n.write(content)
            fall.write("1" + "\t" + content)
        if (rign < 1.5 and lefn >= 1.5):
            fn1.write(content)
            fall.write("2" + "\t" + content)
        if (rign >= 1.5 and lefn >= 1.5):
            fnn.write(content)
            fall.write("3" + "\t" + content)

fall.close()
f11.close()
f1n.close()
fn1.close()
fnn.close()
