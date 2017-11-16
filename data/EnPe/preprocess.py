import random
import xml.etree.ElementTree

ft = open('mytrain.txt', 'w')
fd = open('mydev.txt', 'w')
f_source = open('source.txt', 'a')
f_target = open('target.txt', 'a')
source_lst = []
target_lst = []
f = None
e = xml.etree.ElementTree.parse('train.txt').getroot()
counter = 0
for name in e.findall('Name'):
	source_name = name.find('SourceName')
	counter += 1
	if counter%10==9:
		f = fd
	else:
		f = ft

	for each in list(source_name.text):
		if each not in source_lst:
			source_lst.append(each)

	target = random.choice(name.findall('TargetName'))

	for each in list(target.text):
		if each not in target_lst:
			target_lst.append(each)

	f.write(source_name.text+'\t'+target.text+'\n')

for each in source_lst:
    f_source.write(each+'\n')

for each in target_lst:
    f_target.write(each+'\n')
