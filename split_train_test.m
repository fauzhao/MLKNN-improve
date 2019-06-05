function [train_data,train_target,test_data,test_target] = split_train_test(data,target)
%DATAPROCESS 此处显示有关此函数的摘要
%   此处显示详细说明
    target = target';
    [M,N]=size(data);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices=crossvalind('Kfold',data(1:M,N),10);%进行随机分包
    for k=1:10 %交叉验证k=10，10个包轮流作为测试集
        test = (indices == k); %获得test集元素在数据集中对应的单元编号
        train = ~test;%train集元素的编号为非test元素的编号
        train_data=data(train,:);%从数据集中划分出train样本的数据
        train_target=target(:,train);%获得样本集的测试目标，在本例中是实际分类情况
        test_data=data(test,:);%test样本集
        test_target=target(:,test);
        [HammingLoss(1,k),RankingLoss(1,k),OneError(1,k),Coverage(1,k),Average_Precision(1,k),Outputs,Pre_Labels.MLKNN]=MLKNN_Algorithm(train_data,train_target,test_data,test_target);%要验证的算法
    end
%上述结果为输出算法MLKNN的几个验证指标及最后一轮验证的输出和结果矩阵，每个指标都是一个k元素的行向量
    end

