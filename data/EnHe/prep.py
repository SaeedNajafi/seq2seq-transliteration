import xml.etree.ElementTree

fw = open('dev.txt', 'w')
e = xml.etree.ElementTree.parse('dev.xml').getroot()

for name in e.findall('Name'):
    source_name = name.find('SourceName')
    sc = source_name.text
    sc += " "
    for target in name.findall('TargetName'):
        tg = target.text
        sc += tg + " "

    fw.write(sc.lower().encode('utf8')+'\n')
