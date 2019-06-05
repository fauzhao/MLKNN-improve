    clc;
    clear;
    load('yeast.mat');
    target = targets';
    [M,N]=size(data);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices=crossvalind('Kfold',data(1:M,N),10);%进行随机分包
    para_k = 14;
    para_p = 2;
    para_a = 0.75;
    for k=1:10 %交叉验证k=10，10个包轮流作为测试集
        test = (indices == k); %获得test集元素在数据集中对应的单元编号
        train = ~test;%train集元素的编号为非test元素的编号
        train_data=data(train,:);%从数据集中划分出train样本的数据
        train_target=target(:,train);%获得样本集的测试目标，在本例中是实际分类情况
        test_data=data(test,:);%test样本集
        test_target=target(:,test);
        [HammingLoss(1,k),RankingLoss(1,k),OneError(1,k),Coverage(1,k),Average_Precision(1,k),Outputs,Pre_Labels.MLKNN]=MLKNN_Algorithm(train_data,train_target,test_data,test_target,para_k,1,para_p);%要验证的算法
        [HammingLoss2(1,k),RankingLoss2(1,k),OneError2(1,k),Coverage2(1,k),Average_Precision2(1,k),Outputs2,Pre_Labels.IMLKNN]=IMLKNN_Algorithm(train_data,train_target,test_data,test_target,para_k,1,para_p);
        [HammingLoss3(1,k),RankingLoss3(1,k),OneError3(1,k),Coverage3(1,k),Average_Precision3(1,k),Outputs3,Pre_Labels.CMLKNN]=CMLKNN_Algorithm(train_data,train_target,test_data,test_target,para_k,1,para_a,para_p);
    end
    Name = {'CMLKNN';'IMLKNN';'MLKNN'};
    hloss = {mean(HammingLoss);mean(HammingLoss2);mean(HammingLoss3)};
    OneError = {mean(OneError);mean(OneError2);mean(OneError3)};
    Coverage = {mean(Coverage);mean(Coverage2);mean(Coverage3)};
    rloss = {mean(RankingLoss);mean(RankingLoss2);mean(RankingLoss3)};
    AveragePrecision = {mean(Average_Precision);mean(Average_Precision2);mean(Average_Precision3)};
    table(hloss,OneError,Coverage,rloss,AveragePrecision,'RowNames',Name)

    