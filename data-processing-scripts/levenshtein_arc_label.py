#!/usr/bin/env python3

import gzip
import utils
import numpy as np
from cn_preprocess import CN
import math
import argparse

g_cor, g_sub, g_de, g_ins, g_total = 0, 0, 0, 0, 0 

def main():
	parser = argparse.ArgumentParser(description="This script aligns a conf net to a stm ref")
	parser.add_argument("--stm", dest="stm_file", required=True, help="stm file")
	parser.add_argument("--conf_file", dest="conf_file", required=True, help=".scf.gz")
	parser.add_argument("--npconf_file", dest="npconf_file", required=True, help=".npz")
	args = parser.parse_args()

	stm = args.stm_file
	confusion_net = args.conf_file
	np_conf_net_file = args.npconf_file
	levenshtein_tagging(stm, confusion_net, np_conf_net_file)
		
	return 0

def levenshtein_tagging(stm_file, conf_net_file, np_conf_net_file):
	stm = np.load(stm_file)
	confusion_net = CN(conf_net_file, ignore_graphemes=True)
	arcs = confusion_net.cn_arcs

	start_times=[]
	for arc in arcs:
		start_times.append(arc[1])
	start_times.sort()
	
	#print("np_conf_file", np_conf_net_file)
	np_conf = np.load(np_conf_net_file)
	topo_order = np_conf["topo_order"][:-1] # all but last element as only need parent nodes
	parent_2_child = np_conf["parent_2_child"][()]
	
	# generate conf net sequence, keeping .npz topo order
	# for each sausage have [[word1, log post 1, arc num], [word2, ...], ...]
	extract_one_best=False
	conf_seqs=[]
	for node in topo_order:
		conf_seqs.append([])
		child_dict = parent_2_child[node]
		for key in child_dict: # for conf nets child dict has only one key
			if extract_one_best:
				best_conf=-float('inf')
				arc_data=[]
				for arc in child_dict[key]:
					if arcs[arc][3] > best_conf and arcs[arc][0] != "!NULL":
						best_conf=arcs[arc][3]					
						arc_data=[arcs[arc][0], arcs[arc][3], arc]
				conf_seqs[-1].append(arc_data)
			else:
				for arc in child_dict[key]:
					conf_seqs[-1].append([arcs[arc][0], arcs[arc][3], arc])

	# generate stm sequence	
	start_frame = conf_net_file.split('/')[-1].split('.')[0].split("_")[-2]
	start_frame = float(start_frame)/100
	#start = stm["time"][0]+start_frame
	start = start_times[0]+start_frame
	end = start_times[-1]+start_frame
	stm_seq = get_stm_sequence(stm, start, end)

	#print("stm seq has length", len(stm_seq))
	#print("confnet seq has length", len(conf_seqs))
	

	# compute alignment and generate tags
	scores, traceback = score_matrix(stm_seq, conf_seqs)
	align_stm, align_conf = aligned_seq(stm_seq, conf_seqs, traceback)
	# for i in range(len(align_stm)):
	# 	print("\n", align_stm[i], align_conf[i], "\n")
	tags = tagging(align_stm, align_conf)

	#print(align_stm)
	#print(align_conf)

	return tags

def score_matrix(a, b):
	#fill score matrix and traceback matrix
	scores = np.zeros((len(a)+1, len(b)+1))
	traceback = np.zeros((len(a)+1, len(b)+1))
	
	for j in range(1, len(b)+1):
		scores[0][j] = scores[0][j-1] + insertion(b[j-1])
		traceback[0][j] = -1
	for i in range(1, len(a)+1):
		scores[i][0] = scores[i-1][0] + deletion(a[i-1])
		traceback[i][0] = 1
		for j in range(1, len(b)+1):
			scoreSub = scores[i-1][j-1] + substitution(a[i-1], b[j-1])
			scoreDel = scores[i-1][j] + deletion(a[i-1])
			scoreIns = scores[i][j-1] + insertion(b[j-1])
			scores[i][j] = max(scoreSub, scoreDel, scoreIns)
			if scores[i][j] == scoreSub:
				traceback[i][j] = 0
			elif scores[i][j] == scoreDel:
				traceback[i][j] = 1
			elif scores[i][j] == scoreIns:
				traceback[i][j] = -1
	for i in range(len(a)+1):
		for j in range(len(b)+1):
			scores[i][j]=scores[i][j]
	# print("scores\n",scores)
	# print("traceback\n",traceback)
	return scores, traceback

def aligned_seq(a, b, traceback):
	align_a=[]
	align_b=[]
	i, j = len(a), len(b)
	while i > 0 or j > 0:
		if traceback[i][j] == 0:
			align_a.append(a[i-1])
			align_b.append(b[j-1])
			i-=1
			j-=1
		elif traceback[i][j] == 1:
			align_a.append(a[i-1])
			align_b.append("-")
			i-=1
		else:
			align_a.append("-")
			align_b.append(b[j-1])
			j-=1

	align_a=list(reversed(align_a))
	align_b=list(reversed(align_b))
	return align_a, align_b


#cost functions
def match_score(a, b):#assumes a is a word, b is a list of tuples (word, log arc posterior, arc num)
	match_score = -1
	one_match = False
	for tup in b: # cycle through arcs between nodes
		if tup[0] == a:
			match_score+= math.exp(tup[1])
			one_match = True
	if not one_match:
		match_score = -1
	return match_score

def insertion(a):
	return -1

def deletion(a):
	return -1

def substitution(a, b):
	return match_score(a, b)

#tagging of aligned sequences
def tagging(align_a, align_b):
	tags = {} # key: arc num value: tag
	for i in range(len(align_a)):
		if align_b[i] == "-":
			continue
		else:
			for j in range(len(align_b[i])):
				if align_b[i][j][0] == align_a[i]:
					tags[align_b[i][j][2]] = 1
				else:
					tags[align_b[i][j][2]] = 0
	return tags

def tag_one_best(align_a, align_b):
	tags = []
	for i in range(len(align_b)):
		align_b[i]=align_b[i][0][0]
	cor, sub, de, ins = 0,0,0,0
	for a, b in zip(align_a, align_b):
		#print(a,b)
		if a == b:
			cor+=1
			tags.append("C")
		elif a == "-":
			ins+=1
			tags.append("I")
		elif b == "-":
			de+=1
			tags.append("D")
		else:	
			sub+=1
			tags.append("S")
	
	return cor, sub, de, ins

def get_stm_sequence(stm, start, end):
	# looks at start time of conf net
	best_start, best_end=float('inf'), float('inf')
	best_index_start, best_index_end = 0, 0

	#replace this search by a standard function, this will take forever...
	for i in range(len(stm['time'])):
		diff_start = abs(stm['time'][i] - start)
		diff_end = abs(stm['time'][i] -  end)
		if diff_start < best_start:
			best_start = diff_start
			best_index_start = i
		if diff_end < best_end:
			best_end = diff_end
			best_index_end = i
	
	stm_seq=[]
	for i in range(best_index_start, best_index_end+1):
		stm_seq.append(stm['word'][i])
	return stm_seq

def get_one_best(a):
	one_best=[]
	for sau in a:
		best_conf=-float('inf')
		word=''
		for arc in sau:
			if arc[1] > best_conf:
				best_conf=arc[1]
				word = arc[0]
		one_best.append(word)
	return one_best

if __name__ == '__main__':
    main()
