import xml.etree.ElementTree

pred = open('test.predicted', 'r')
e = xml.etree.ElementTree.parse('./data/EnPe/dev.txt').getroot()
total_source = 0.0
total_correct = 0.0
preds = pred.readlines()
index = 0
for name in e.findall('Name'):
    pred_target = preds[index]
    total_source += 1

	for target in name.findall('TargetName'):
        if target.decode('utf8') == pred_target.decode('utf8'):
            total_correct += 1
            break

    index += 1

print (total_correct/total_source)*100
