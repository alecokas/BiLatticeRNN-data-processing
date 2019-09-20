import os

# DIR = '/home/babel/BABEL_OP3_404/releaseB/exp-graphemic-pmn26/J2/decode-ibmseg-fcomb/test/dev'
DIR = '/home/babel/BABEL_OP2_202/releaseB/exp-graphemic-ar527-v3/J1/decode-ibmseg-fcomb/test/dev/'
TGT = 'swahili.dev.lst'
EXTENSION = '.scf.gz'

names_set = set()
for root, dirs, names in os.walk(DIR):
    for name in names:
        if name.endswith(EXTENSION):
            names_set.add(name[:-len(EXTENSION)])

with open(TGT, 'w') as tgt_file:
    tgt_file.write('\n'.join(list(names_set)))
