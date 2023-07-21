'''
References
https://www.analyticsvidhya.com/blog/2020/10/all-about-decision-tree-from-scratch-with-python-implementation/
https://betterdatascience.com/mml-decision-trees/
https://levelup.gitconnected.com/building-a-decision-tree-from-scratch-in-python-machine-learning-from-scratch-part-ii-6e2e56265b19
'''


# main library 
import numpy as np

def main():
    print('START Q1_AB\n')
    
    #  reading training data
    data_train_file = open(file="datasets/Q1_train.txt", mode="r")
    data_train_file_content = data_train_file.readlines()
    final_training_data = []
    for loop_file_data in data_train_file_content:
        loop_file_data = loop_file_data.replace("(", "")
        loop_file_data = loop_file_data.replace(")", "")
        loop_file_data = loop_file_data.replace(" ", "")
        loop_file_data = loop_file_data.replace("\n", "")
        loop_file_data = loop_file_data.split(",")
        loop_file_data[0] = float(loop_file_data[0])
        loop_file_data[1] = float(loop_file_data[1])
        loop_file_data[2] = int(loop_file_data[2])
        loop_file_data[3] = str(loop_file_data[3])
        final_training_data.append(loop_file_data)

    # reading testing data
    File_Test = open(file="datasets/Q1_test.txt", mode="r")
    File_Test_Content = File_Test.readlines()
    final_testing_data = []
    for loop_file_data in File_Test_Content:
        loop_file_data = loop_file_data.replace("(", "")
        loop_file_data = loop_file_data.replace(")", "")
        loop_file_data = loop_file_data.replace(" ", "")
        loop_file_data = loop_file_data.replace("\n", "")
        loop_file_data = loop_file_data.split(",")
        loop_file_data[0] = float(loop_file_data[0])
        loop_file_data[1] = float(loop_file_data[1])
        loop_file_data[2] = int(loop_file_data[2])
        loop_file_data[3] = str(loop_file_data[3])
        final_testing_data.append(loop_file_data)

    # reading train features
    final_training_featuress = []
    for loop_file_data in final_training_data:
        loop_featuress_data = loop_file_data[:3]
        final_training_featuress.append(loop_featuress_data)

    # reading train labels
    final_training_labels = []
    final_training_labels_vector = []
    for loop_file_data in final_training_data:
        loop_labelss_data = loop_file_data[3]
        final_training_labels.append(loop_labelss_data)
        final_training_labels_vector.append(1 if loop_labelss_data == 'M' else 0)

    # reading test features
    final_testing_featuress = []
    for loop_file_data in final_testing_data:
        loop_featuress_data = loop_file_data[:3]
        final_testing_featuress.append(loop_featuress_data)

    # reading test labels
    final_testing_labelss = []
    final_testing_labels_vector = []
    for loop_file_data in final_testing_data:
        loop_labelss_data = loop_file_data[3]
        final_testing_labelss.append(loop_labelss_data)
        final_testing_labels_vector.append(1 if loop_labelss_data == 'M' else 0)

    # np conversions
    final_training_data = np.array(final_training_data)
    final_training_featuress = np.array(final_training_featuress)
    final_training_labels = np.array(final_training_labels)
    final_training_labels_vector = np.array(final_training_labels_vector)
    final_testing_data = np.array(final_testing_data)
    final_testing_featuress = np.array(final_testing_featuress)
    final_testing_labelss = np.array(final_testing_labelss)
    final_testing_labels_vector = np.array(final_testing_labels_vector)
    
    # main depth run
    for loop_depth_value in range(1,6):
        decision_tree_trained_model = class_decision_tree(cls_maxi_depthh = loop_depth_value)
        final_training_featuress = np.array(final_training_featuress)
        final_training_labels_vector = np.array(final_training_labels_vector)
        decision_tree_trained_model.main_fit(final_training_featuress, final_training_labels_vector)
        prediction_for_training_data = decision_tree_trained_model.main_predict(final_training_featuress)
        accuracy_for_training_data = func_acc_score_cal(final_training_labels_vector, prediction_for_training_data)
        final_testing_featuress = np.array(final_testing_featuress)
        prediction_for_testing_data = decision_tree_trained_model.main_predict(final_testing_featuress)
        accuracy_for_testing_data = func_acc_score_cal(final_testing_labels_vector, prediction_for_testing_data)
        print("DEPTH =", loop_depth_value)
        print("Accuraccy | Train =", accuracy_for_training_data, end="")
        print(" | Test =", accuracy_for_testing_data)

    print('END Q1_AB\n')

# finding mode values
def func_mode_value_cal(ins_data):
    ins_word_frequency_dict = {}
    for word in ins_data:
        if word in ins_word_frequency_dict:
            ins_word_frequency_dict[word] += 1
        else:
            ins_word_frequency_dict[word] = 1
    final_sorted_data = sorted(ins_word_frequency_dict, key = ins_word_frequency_dict.get, reverse = True)
    return_results = final_sorted_data[0]
    return return_results

# function to find accuracy
def func_acc_score_cal(ins_real_data, ins_pred_data):
    ins_correct_data = [1 for i in range(len(ins_real_data)) if ins_real_data[i] == ins_pred_data[i] ]
    ins_correct_data = ins_correct_data.count(1)
    ins_final_accuracy = ins_correct_data / float(len(ins_real_data)) 
    return round(ins_final_accuracy, 1)

# node class for main tre nodes
class class_main_node:
    def __init__(self, ins_feature=None, ins_thresh=None, ins_left_data=None, ins_ryt_data=None, ins_gain_val=None, ins_node_value=None):
        self.ins_feature = ins_feature
        self.ins_thresh = ins_thresh
        self.ins_left_data = ins_left_data
        self.ins_ryt_data = ins_ryt_data
        self.ins_gain_val = ins_gain_val
        self.ins_node_value = ins_node_value

# decision tree class
class class_decision_tree:
    
    # depth setting function
    def __init__(self, cls_maxi_depthh=5):
        self.cls_maxi_depthh = cls_maxi_depthh
        self.cls_root = None
        
    
    # main tree build function
    def func_building_tree(self, instance_featuress, instance_labelss, internal_depth_values=0):
        ins_no_of_rows, ins_no_of_columns = instance_featuress.shape
        if ins_no_of_rows >= self.cls_min_split and internal_depth_values <= self.cls_maxi_depthh:
            
            # finding best split
            ins_best_split_dict = {}
            best_info_gain = -1
            ins_no_of_rows, ins_no_of_columns = instance_featuress.shape
            for main_loop_index in range(ins_no_of_columns):
                ins_current_fea_val = instance_featuress[:, main_loop_index]
                for ins_thresh in np.unique(ins_current_fea_val):
                    ins_dataframe = np.concatenate((instance_featuress, instance_labelss.reshape(1, -1).T), axis=1)
                    ins_df_left = np.array([row for row in ins_dataframe if row[main_loop_index] <= ins_thresh])
                    ins_df_ryt = np.array([row for row in ins_dataframe if row[main_loop_index] > ins_thresh])
                    if len(ins_df_left) > 0 and len(ins_df_ryt) > 0:
                        instance_labelss = ins_dataframe[:, -1]
                        
                        # finding information ins_gain_val
                        parent, left_child, right_child = instance_labelss, ins_df_left[:, -1], ins_df_ryt[:, -1]
                        num_left = len(left_child) / len(parent)
                        num_right = len(right_child) / len(parent)
                        
                        # entropy calculations parent
                        s_parent = parent
                        counts_parent = np.bincount(np.array(s_parent, dtype=np.int64))
                        entropy_parent = 0
                        for pct_parent in (counts_parent/len(s_parent)):
                            if pct_parent > 0:
                                entropy_parent += pct_parent * np.log2(pct_parent)
                        parent_entropy = -entropy_parent
                        
                        # entropy calculations left
                        s_left_child = left_child
                        counts_left_child = np.bincount(np.array(s_left_child, dtype=np.int64))
                        entropy_left_child = 0
                        for pct_left_child in (counts_left_child/len(s_left_child)):
                            if pct_left_child > 0:
                                entropy_left_child += pct_left_child * np.log2(pct_left_child)
                        left_entropy = -entropy_left_child
                        
                        # entropy calculations left
                        s_right_child = right_child
                        counts_right_child = np.bincount(np.array(s_right_child, dtype=np.int64))
                        entropy_right_child = 0
                        for pct_right_child in (counts_right_child/len(s_right_child)):
                            if pct_right_child > 0:
                                entropy_right_child += pct_right_child * np.log2(pct_right_child)
                        right_entropy = -entropy_right_child
                        
                        # finding final ins_gain_val
                        ins_gain_val = parent_entropy - (num_left * left_entropy + num_right * right_entropy)
                        
                        # main information ins_gain_val comparison
                        if ins_gain_val > best_info_gain:
                            ins_best_split_dict = { 'ins_idx_val': main_loop_index, 'ins_thresh': ins_thresh, 'ins_df_left': ins_df_left, 'ins_df_ryt': ins_df_ryt, 'ins_gain_val': ins_gain_val }
                            best_info_gain = ins_gain_val
            
            # saving best split
            found_best_split = ins_best_split_dict
            
            # comparison
            if found_best_split['ins_gain_val'] > 0:
                ins_left_ans = self.func_building_tree( instance_featuress=found_best_split['ins_df_left'][:, :-1], instance_labelss=found_best_split['ins_df_left'][:, -1],  internal_depth_values=internal_depth_values + 1 )
                ins_ryt_ans = self.func_building_tree( instance_featuress=found_best_split['ins_df_ryt'][:, :-1], instance_labelss=found_best_split['ins_df_ryt'][:, -1], internal_depth_values=internal_depth_values + 1 )
                return class_main_node( ins_feature=found_best_split['ins_idx_val'], ins_thresh=found_best_split['ins_thresh'], ins_left_data=ins_left_ans, ins_ryt_data=ins_ryt_ans, ins_gain_val=found_best_split['ins_gain_val'] )
        
        # returning mode
        return class_main_node( ins_node_value=func_mode_value_cal(instance_labelss) )
    
    # recursive function for iteratign tree
    def func_predict(self, cls_unit_feature, cls_tree):
        if cls_tree.ins_node_value != None:
            return cls_tree.ins_node_value
        instance_features_values = cls_unit_feature[cls_tree.ins_feature]
        if instance_features_values <= cls_tree.ins_thresh:
            return self.func_predict(cls_unit_feature=cls_unit_feature, cls_tree=cls_tree.ins_left_data)
        if instance_features_values > cls_tree.ins_thresh:
            return self.func_predict(cls_unit_feature=cls_unit_feature, cls_tree=cls_tree.ins_ryt_data)

    # fit method
    def main_fit(self, instance_featuress, instance_labelss):
        self.cls_min_split = 2
        self.cls_root = self.func_building_tree(instance_featuress, instance_labelss)

    # predict function
    def main_predict(self, instance_featuress):
        main_prediction_list = []
        for loop_feature_pick in instance_featuress:
            instant_results = self.func_predict(loop_feature_pick, self.cls_root)
            main_prediction_list.append(instant_results)
        return main_prediction_list

if __name__ == "__main__":
    main()
