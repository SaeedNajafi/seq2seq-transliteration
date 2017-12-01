import sys

pred_lines = open(sys.argv[1], 'r').readlines()
ref_lines = open(sys.argv[2], 'r').readlines()
total_source = 0.0
total_correct = 0.0
index = 0
for ref in ref_lines:
    pred_target = pred_lines[index].strip()
    total_source += 1
    ref_tg_list = ref.strip().split(" ")[1:]
    for target in ref_tg_list:
        if target == pred_target:
            total_correct += 1
            break
    index += 1

print ((total_correct/total_source)*100)
