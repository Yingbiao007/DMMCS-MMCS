import os
import torch

data_root_path = 'E:\\thesis_for_me\\thesis_codes\\DMMCS\\data'


def get_N_M_dict(data_path):
    with open(data_path, 'r', encoding='utf-8') as file_in:
        head_rel_dic = {}
        tail_rel_dict = {}
        head_rel_number = {}
        tail_rel_number = {}
        for line in file_in:
            line = str(line).strip().split('\t')
            head, relation, tail = line
            if (head, relation) not in head_rel_dic.keys():
                head_rel_dic[(head, relation)] = []
                head_rel_number[(head, relation)] = 0
            if (tail, relation) not in tail_rel_dict.keys():
                tail_rel_dict[(tail, relation)] = []
                tail_rel_number[(tail, relation)] = 0
            head_rel_dic[(head, relation)].append(tail)
            head_rel_number[(head, relation)] += 1
            tail_rel_dict[(tail, relation)].append(head)
            tail_rel_number[(tail, relation)] += 1
    return head_rel_dic, tail_rel_dict, head_rel_number, tail_rel_number


if __name__ == '__main__':
    data = input('data_name:\n')
    data_name = data + '\\train.txt'
    rel_name = data + '\\relations.dict'
    data_path = os.path.join(data_root_path, data_name)
    rel_path = os.path.join(data_root_path, rel_name)
    head_rel_dic, tail_rel_dict, head_rel_number, tail_rel_number = get_N_M_dict(data_path)
    tail_rel_number = sorted(tail_rel_number.items(), key=lambda item: item[1], reverse=True)
    with open('D:\\python_project\\test_data\\yago_tail_rel.txt','w',encoding='utf-8') as file_out:
        for item in tail_rel_number:
            file_out.writelines(str(item)+'\n')
    print(len(tail_rel_dict.keys()))