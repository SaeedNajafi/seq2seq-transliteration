import xml.etree.ElementTree

mapping = {}
mapping[u'\ufe8d'] = u'\u0627'
mapping[u'\ufe8e'] = u'\u0627'
mapping[u'\u0627'] = u'\u0627'

mapping[u'\ufe8f'] = u'\u0628'
mapping[u'\ufe90'] = u'\u0628'
mapping[u'\ufe92'] = u'\u0628'
mapping[u'\ufe91'] = u'\u0628'
mapping[u'\u0628'] = u'\u0628'

mapping[u'\ufe95'] = u'\u062a'
mapping[u'\ufe96'] = u'\u062a'
mapping[u'\ufe97'] = u'\u062a'
mapping[u'\ufe98'] = u'\u062a'
mapping[u'\u062a'] = u'\u062a'

mapping[u'\ufe99'] = u'\u062b'
mapping[u'\ufe9a'] = u'\u062b'
mapping[u'\ufe9b'] = u'\u062b'
mapping[u'\ufe9c'] = u'\u062b'
mapping[u'\u062b'] = u'\u062b'

mapping[u'\ufe9d'] = u'\u062c'
mapping[u'\ufe9e'] = u'\u062c'
mapping[u'\ufea0'] = u'\u062c'
mapping[u'\ufe9f'] = u'\u062c'
mapping[u'\u062c'] = u'\u062c'

mapping[u'\ufea1'] = u'\u062d'
mapping[u'\ufea2'] = u'\u062d'
mapping[u'\ufea3'] = u'\u062d'
mapping[u'\ufea4'] = u'\u062d'
mapping[u'\u062d'] = u'\u062d'

mapping[u'\ufea5'] = u'\u062e'
mapping[u'\ufea6'] = u'\u062e'
mapping[u'\ufea7'] = u'\u062e'
mapping[u'\ufea8'] = u'\u062e'
mapping[u'\u062e'] = u'\u062e'

mapping[u'\ufea9'] = u'\u062f'
mapping[u'\ufeaa'] = u'\u062f'
mapping[u'\u062f'] = u'\u062f'

mapping[u'\ufeab'] = u'\u0630'
mapping[u'\ufeac'] = u'\u0630'
mapping[u'\u0630'] = u'\u0630'

mapping[u'\ufead'] = u'\u0631'
mapping[u'\ufeae'] = u'\u0631'
mapping[u'\u0631'] = u'\u0631'

mapping[u'\ufeaf'] = u'\u0632'
mapping[u'\ufeb0'] = u'\u0632'
mapping[u'\u0632'] = u'\u0632'

mapping[u'\ufeb1'] = u'\u0633'
mapping[u'\ufeb2'] = u'\u0633'
mapping[u'\ufeb3'] = u'\u0633'
mapping[u'\ufeb4'] = u'\u0633'
mapping[u'\u0633'] = u'\u0633'

mapping[u'\ufeb5'] = u'\u0634'
mapping[u'\ufeb6'] = u'\u0634'
mapping[u'\ufeb7'] = u'\u0634'
mapping[u'\ufeb8'] = u'\u0634'
mapping[u'\u0634'] = u'\u0634'

mapping[u'\ufeb9'] = u'\u0635'
mapping[u'\ufeba'] = u'\u0635'
mapping[u'\ufebb'] = u'\u0635'
mapping[u'\ufebc'] = u'\u0635'
mapping[u'\u0635'] = u'\u0635'

mapping[u'\ufebd'] = u'\u0636'
mapping[u'\ufebe'] = u'\u0636'
mapping[u'\ufebf'] = u'\u0636'
mapping[u'\ufec0'] = u'\u0636'
mapping[u'\u0636'] = u'\u0636'

mapping[u'\ufec1'] = u'\u0637'
mapping[u'\ufec2'] = u'\u0637'
mapping[u'\ufec3'] = u'\u0637'
mapping[u'\ufec4'] = u'\u0637'
mapping[u'\u0637'] = u'\u0637'


mapping[u'\ufec5'] = u'\u0638'
mapping[u'\ufec6'] = u'\u0638'
mapping[u'\ufec7'] = u'\u0638'
mapping[u'\ufec8'] = u'\u0638'
mapping[u'\u0638'] = u'\u0638'


mapping[u'\ufec9'] = u'\u0639'
mapping[u'\ufeca'] = u'\u0639'
mapping[u'\ufecb'] = u'\u0639'
mapping[u'\ufecc'] = u'\u0639'
mapping[u'\u0639'] = u'\u0639'

mapping[u'\ufecd'] = u'\u063a'
mapping[u'\ufece'] = u'\u063a'
mapping[u'\ufecf'] = u'\u063a'
mapping[u'\ufed0'] = u'\u063a'
mapping[u'\u063a'] = u'\u063a'

mapping[u'\ufed1'] = u'\u0641'
mapping[u'\ufed2'] = u'\u0641'
mapping[u'\ufed2'] = u'\u0641'
mapping[u'\ufed4'] = u'\u0641'
mapping[u'\u0641'] = u'\u0641'

mapping[u'\ufed5'] = u'\u0642'
mapping[u'\ufed6'] = u'\u0642'
mapping[u'\ufed7'] = u'\u0642'
mapping[u'\ufed8'] = u'\u0642'
mapping[u'\u0642'] = u'\u0642'

mapping[u'\ufed9'] = u'\u0643'
mapping[u'\ufeda'] = u'\u0643'
mapping[u'\ufedb'] = u'\u0643'
mapping[u'\ufedc'] = u'\u0643'
mapping[u'\u0643'] = u'\u0643'
mapping[u'\u06a9'] = u'\u0643'

mapping[u'\ufedd'] = u'\u0644'
mapping[u'\ufede'] = u'\u0644'
mapping[u'\ufedf'] = u'\u0644'
mapping[u'\ufee0'] = u'\u0644'
mapping[u'\u0644'] = u'\u0644'

mapping[u'\ufee1'] = u'\u0645'
mapping[u'\ufee2'] = u'\u0645'
mapping[u'\ufee3'] = u'\u0645'
mapping[u'\ufee4'] = u'\u0645'
mapping[u'\u0645'] = u'\u0645'

mapping[u'\ufee5'] = u'\u0646'
mapping[u'\ufee6'] = u'\u0646'
mapping[u'\ufee7'] = u'\u0646'
mapping[u'\ufee8'] = u'\u0646'
mapping[u'\u0646'] = u'\u0646'

mapping[u'\ufee9'] = u'\u0647'
mapping[u'\ufeea'] = u'\u0647'
mapping[u'\ufeeb'] = u'\u0647'
mapping[u'\ufeec'] = u'\u0647'
mapping[u'\u0647'] = u'\u0647'

mapping[u'\ufeed'] = u'\u0648'
mapping[u'\ufeee'] = u'\u0648'
mapping[u'\u0648'] = u'\u0648'

mapping[u'\ufef1'] = u'\u064a'
mapping[u'\ufef2'] = u'\u064a'
mapping[u'\ufef3'] = u'\u064a'
mapping[u'\ufef4'] = u'\u064a'
mapping[u'\u064a'] = u'\u064a'

mapping[u'\ufe81'] = u'\u0622'
mapping[u'\ufe82'] = u'\u0622'
mapping[u'\u0622'] = u'\u0622'

mapping[u'\ufe93'] = u'\u0629'
mapping[u'\ufe94'] = u'\u0629'
mapping[u'\u0629'] = u'\u0629'

mapping[u'\ufeef'] = u'\u0649'
mapping[u'\ufef0'] = u'\u0649'
mapping[u'\u0649'] = u'\u0649'
mapping[u'\u06cc'] = u'\u0649'

mapping[u'\u0686'] = u'\u0686'

mapping[u'\u06af'] = u'\u06af'

mapping[u'\u067e'] = u'\u067e'

mapping[u'\u0698'] = u'\u0698'

fw = open('corrected_train.txt', 'w')
e = xml.etree.ElementTree.parse('train.xml').getroot()

for name in e.findall('Name'):
    source_name = name.find('SourceName')
    sc = source_name.text
    sc += " "
    for target in name.findall('TargetName'):
        tg = target.text
	#tg = "".join([mapping[e] for e in list(target.text)])
        sc += tg + " "

    fw.write(sc.encode('utf8')+'\n')
