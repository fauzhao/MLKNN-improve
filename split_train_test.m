function [train_data,train_target,test_data,test_target] = split_train_test(data,target)
%DATAPROCESS �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    target = target';
    [M,N]=size(data);%���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
    indices=crossvalind('Kfold',data(1:M,N),10);%��������ְ�
    for k=1:10 %������֤k=10��10����������Ϊ���Լ�
        test = (indices == k); %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
        train = ~test;%train��Ԫ�صı��Ϊ��testԪ�صı��
        train_data=data(train,:);%�����ݼ��л��ֳ�train����������
        train_target=target(:,train);%����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
        test_data=data(test,:);%test������
        test_target=target(:,test);
        [HammingLoss(1,k),RankingLoss(1,k),OneError(1,k),Coverage(1,k),Average_Precision(1,k),Outputs,Pre_Labels.MLKNN]=MLKNN_Algorithm(train_data,train_target,test_data,test_target);%Ҫ��֤���㷨
    end
%�������Ϊ����㷨MLKNN�ļ�����ָ֤�꼰���һ����֤������ͽ������ÿ��ָ�궼��һ��kԪ�ص�������
    end

