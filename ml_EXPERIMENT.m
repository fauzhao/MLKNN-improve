    clc;
    clear;
    load('yeast.mat');
    target = targets';
    [M,N]=size(data);%���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
    indices=crossvalind('Kfold',data(1:M,N),10);%��������ְ�
    para_k = 14;
    para_p = 2;
    para_a = 0.75;
    for k=1:10 %������֤k=10��10����������Ϊ���Լ�
        test = (indices == k); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
        train = ~test;%train��Ԫ�صı��Ϊ��testԪ�صı��
        train_data=data(train,:);%�����ݼ��л��ֳ�train����������
        train_target=target(:,train);%����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
        test_data=data(test,:);%test������
        test_target=target(:,test);
        [HammingLoss(1,k),RankingLoss(1,k),OneError(1,k),Coverage(1,k),Average_Precision(1,k),Outputs,Pre_Labels.MLKNN]=MLKNN_Algorithm(train_data,train_target,test_data,test_target,para_k,1,para_p);%Ҫ��֤���㷨
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

    