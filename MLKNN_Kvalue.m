    clc;
    clear;
    load('yeast.mat');
    target = targets';
    [M,N]=size(data);%���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
    indices=crossvalind('Kfold',data(1:M,N),10);%��������ְ�
    para_k = 5;
    para_p = 1;
    i=1;      %iΪ1ʱ��ӦkΪ5 ���� k=i+4
    hloss = [];
    rloss = [];
    while (para_k < 26)
        for k=1:10 %������֤k=10��10����������Ϊ���Լ�
            test = (indices == k); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
            train = ~test;%train��Ԫ�صı��Ϊ��testԪ�صı��
            train_data=data(train,:);%�����ݼ��л��ֳ�train����������
            train_target=target(:,train);%����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
            test_data=data(test,:);%test������
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
    
    
    