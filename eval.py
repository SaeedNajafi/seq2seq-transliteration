import xml.etree.ElementTree

pred = open('test.predicted', 'r')
e = xml.etree.ElementTree.parse('./data/EnPe/dev.xml').getroot()
total_source = 0.0
total_correct = 0.0
preds = pred.readlines()
index = 0
for name in e.findall('Name'):
    pred_target = preds[index].strip()
    total_source += 1

    for target in name.findall('TargetName'):
        print(list(target.text))
        print(list(pred_target))
        print(list(target.text.encode('utf-8')))
        print(list(pred_target.encode('utf-8')))
        if target.text.strip() == pred_target.strip():
            print("TRUE")
            total_correct += 1
            break
        else:
            print ("FALSE")

    index += 1
print ((total_correct/total_source)*100)
