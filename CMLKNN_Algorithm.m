function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = CMLKNN_Algorithm(train_data,train_target,test_data,test_target,K,Smooth,a,para_p)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
    %[train_data,train_target,test_data,test_target] = split_train_test(data,target)
    [Prior,PriorN,Cond,CondN,maxF,maxFN,m,n] = CMLKNN_train(train_data,train_target,K,Smooth,para_p);
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=CMLKNN_test(train_data,train_target,test_data,test_target,K,Prior,PriorN,Cond,CondN,maxF,maxFN,m,n,a);
end
