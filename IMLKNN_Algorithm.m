function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = IMLKNN_Algorithm(train_data,train_target,test_data,test_target,K,Smooth,para_p)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    %[train_data,train_target,test_data,test_target] = split_train_test(data,target)
    [Prior,PriorN,Cond,CondN] = IMLKNN_train(train_data,train_target,K,Smooth,para_p);
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=IMLKNN_test(train_data,train_target,test_data,test_target,K,Prior,PriorN,Cond,CondN);
end

