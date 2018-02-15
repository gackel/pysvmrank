#coding: utf-8
import os
import sys
from tempfile import NamedTemporaryFile

class SVMRank:
	def __init__(self, MODEL_NAME="model"):
		self.MODEL_NAME = MODEL_NAME
		self.train_data_list = []
		self.test_data_list = []

	def add_train_data(self, feature_dict_list, label_list):
		"""
		qid ごとに呼ばれる
		feature_dict_list: [{1:0.3, 2:0.7}, ..]
		label_list: [3, 2, 1, 1, ..]
			-- feature_dict は key: int, value: float (int)
			-- label_list は float (int) の必要あり
		"""
		assert len(feature_dict_list) == len(label_list)
		tmplist = []
		for feature_dict, label in zip(feature_dict_list, label_list):
			tmplist.append((feature_dict, label))
		self.train_data_list.append(tmplist)

	def add_test_data(self, feature_dict_list, ans_label_list):
		assert len(feature_dict_list) == len(ans_label_list)
		tmplist = []
		for feature_dict, label in zip(feature_dict_list, ans_label_list):
			tmplist.append((feature_dict, label))
		self.test_data_list.append(tmplist)


	def clear_train_data(self):
		self.train_data_list = []

	def clear_test_data(self):
		self.test_data_list = []

	def train(self):
		"""
		SVMRankの学習を行う
		"""
		train_file_name = self.write_features(self.train_data_list)
		os.system("svm_rank_learn -c 20.0 %s %s" % (train_file_name, self.MODEL_NAME))
		print("svm_rank_learn -c 20.0 %s %s" % (train_file_name, self.MODEL_NAME))

	def test(self):
		test_file_name = self.write_features(self.test_data_list)

		result_file = NamedTemporaryFile(delete=False)
		os.system("svm_rank_classify %s %s %s" % (test_file_name, self.MODEL_NAME, result_file.name))
		print("svm_rank_classify %s %s %s" % (test_file_name, self.MODEL_NAME, result_file.name))
		result_file.close()

		result_list = []
		qid = 0
		with open(result_file.name) as f:
			while len(self.test_data_list) > qid:
				tmplist = []
				for line in iter(f.readline,""):
					tmplist.append(float(line.strip()))
					if len(tmplist) == len(self.test_data_list[qid]):
						break
				qid += 1
				result_list.append(tmplist)
		return result_list


	def write_features(self, data_list):
		# featureの書き込み
		f = NamedTemporaryFile(delete=False)
		for qid,tmplist in enumerate(data_list):
			for feature_dict, label in tmplist:
                            f.write(b"%d qid:%d %s\n" % (label, qid+1, b" ".join([b"%d:%.3f"%(k,v) for (k,v) in sorted(feature_dict.items())])))
		feature_file_name = f.name
		f.close()
		return feature_file_name


if __name__ == "__main__":
	list_of_train_feature_dict_list = [
		[
			{1:1, 2:1, 3:0, 4:0.2, 5:0},
			{1:0, 2:0, 3:1, 4:0.1, 5:1},
			{1:0, 2:1, 3:0, 4:0.4, 5:0},
			{1:0, 2:0, 3:1, 4:0.3, 5:0}
		],
		[
			{1:0, 2:0, 3:1, 4:0.2, 5:0},
			{1:1, 2:0, 3:1, 4:0.4, 5:0},
			{1:0, 2:0, 3:1, 4:0.1, 5:0},
			{1:0, 2:0, 3:1, 4:0.2, 5:0}
		],
		[
			{1:0, 2:0, 3:1, 4:0.1, 5:1},
			{1:1, 2:1, 3:0, 4:0.3, 5:0},
			{1:1, 2:0, 3:0, 4:0.4, 5:1},
			{1:0, 2:1, 3:1, 4:0.5, 5:0}
		]
	]

	list_of_train_label_list = [
		[3,2,1,1],
		[1,2,1,1],
		[2,3,4,1]
	]

	list_of_test_feature_dict_list = [
		[
			{1:1, 2:1, 3:0, 4:0.3, 5:0},
			{1:0, 2:0, 3:0, 4:0.2, 5:1},
			{1:0, 2:0, 3:1, 4:0.2, 5:0},
			{1:1, 2:0, 3:0, 4:0.2, 5:1}
		],
		[	
			{1:1, 2:1, 3:0, 4:0.3, 5:0},
			{1:0, 2:0, 3:0, 4:0.2, 5:1},
			{1:0, 2:0, 3:1, 4:0.2, 5:0},
			{1:1, 2:0, 3:0, 4:0.2, 5:1}
		]
	]
	list_of_test_label_list = [
		[3,2,1,4],
		[3,2,1,4]
	]


	SR = SVMRank()
	for train_feature_dict_list, train_label_list in zip(list_of_train_feature_dict_list, list_of_train_label_list):
		SR.add_train_data(train_feature_dict_list, train_label_list)
	SR.train()

	for test_feature_dict_list, test_label_list in zip(list_of_test_feature_dict_list, list_of_test_label_list):
		SR.add_test_data(test_feature_dict_list, test_label_list)
	print(SR.test())
