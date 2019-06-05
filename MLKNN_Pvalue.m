    clc;
    clear;
    load('yeast.mat');
    target = targets';
    [M,N]=size(data);%���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
    indices=crossvalind('Kfold',data(1:M,N),10);%��������ְ�
    para_k = 14;
    para_p = 1;
    hloss = [];
    rloss = [];
    one_error = [];
    coverage = [];
    avgprec = [];
    while (para_p < 4)
        for k=1:10 %������֤k=10��10����������Ϊ���Լ�
            test = (indices == k); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
            train = ~test;%train��Ԫ�صı��Ϊ��testԪ�صı��
            train_data=data(train,:);%�����ݼ��л��ֳ�train����������
            train_target=target(:,train);%����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
            test_data=data(test,:);%test������
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
    Name = {'�����پ���';'ŷ����þ���';'�б�ѩ�����'};
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
    