    clc;
    clear;
    load('yeast.mat');
    target = targets';
    [M,N]=size(data);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices=crossvalind('Kfold',data(1:M,N),10);%进行随机分包
    para_k = 5;
    para_p = 1;
    i=1;      %i为1时对应k为5 所以 k=i+4
    hloss = [];
    rloss = [];
    while (para_k < 26)
        for k=1:10 %交叉验证k=10，10个包轮流作为测试集
            test = (indices == k); %获得test集元素在数据集中对应的单元编号
            train = ~test;%train集元素的编号为非test元素的编号
            train_data=data(train,:);%从数据集中划分出train样本的数据
            train_target=target(:,train);%获得样本集的测试目标，在本例中是实际分类情况
            test_data=data(test,:);%test样本集
            test_target=target(:,test);
            [HammingLoss(1,k),RankingLoss(1,k),OneError(1,k),Coverage(1,k),Average_Precision(1,k),Outputs,Pre_Labels.MLKNN]=MLKNN_Algorithm(train_data,train_target,test_data,test_target,para_k,1,para_p);
        end
        %disp(strcat('hloss:',num2str(mean(HammingLoss))));
        hloss(i) = mean(HammingLoss);
        rloss(i) = mean(RankingLoss);
        i= i+1;
        para_k = para_k + 1;
    end
    %plot(hloss);
    %plot(rloss);
    
    
    