    clc;
    clear;
    load('yeast.mat');
    target = targets';
    [M,N]=size(data);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices=crossvalind('Kfold',data(1:M,N),10);%进行随机分包
    para_k = 14;
    para_p = 1;
    hloss = [];
    rloss = [];
    one_error = [];
    coverage = [];
    avgprec = [];
    while (para_p < 4)
        for k=1:10 %交叉验证k=10，10个包轮流作为测试集
            test = (indices == k); %获得test集元素在数据集中对应的单元编号
            train = ~test;%train集元素的编号为非test元素的编号
            train_data=data(train,:);%从数据集中划分出train样本的数据
            train_target=target(:,train);%获得样本集的测试目标，在本例中是实际分类情况
            test_data=data(test,:);%test样本集
            test_target=target(:,test);
            [HammingLoss(1,k),RankingLoss(1,k),OneError(1,k),Coverage(1,k),Average_Precision(1,k),Outputs,Pre_Labels.MLKNN]=MLKNN_Algorithm(train_data,train_target,test_data,test_target,para_k,1,para_p);
        end
        hloss(para_p) = mean(HammingLoss);
        rloss(para_p) = mean(RankingLoss);
        one_error(para_p) = mean(OneError);
        coverage(para_p) = mean(Coverage);
        avgprec(para_p) = mean(Average_Precision); 
        para_p= para_p+1;
    end
    Name = {'曼哈顿距离';'欧几里得距离';'切比雪夫距离'};
    hloss = {hloss(1);hloss(2);hloss(3)};
    OneError = {one_error(1);one_error(1);one_error(1)};
    Coverage = {coverage(1);coverage(2);coverage(3)};
    rloss = {rloss(1);rloss(2);rloss(3)};
    AveragePrecision = {avgprec(1);avgprec(2);avgprec(3)};
    table(hloss,OneError,Coverage,rloss,AveragePrecision,'RowNames',Name)
    
    
    
    
    
    
    
%     disp(strcat('hloss:',num2str(mean(HammingLoss))));
%     disp(strcat('one_error:',num2str(mean(OneError))));
%     disp(strcat('coverage:',num2str(mean(Coverage))));
%     disp(strcat('rloss:',num2str(mean(RankingLoss))));
%     disp(strcat('avgprec:',num2str(mean(Average_Precision))));
    