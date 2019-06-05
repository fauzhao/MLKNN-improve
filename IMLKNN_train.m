function [Prior,PriorN,Cond,CondN,NeighborsTarget]=IMLKNN_train(train_data,train_target,Num,Smooth,para_p)
%MLKNN_train trains a multi-label k-nearest neighbor classifier
%
%    Syntax
%
%       [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,num_neighbor)
%
%    Description
%
%       KNNML_train takes,
%           train_data   - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target - A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1 
%           Num          - Number of neighbors used in the k-nearest neighbor algorithm
%           Smooth       - Smoothing parameter
%      and returns,
%           Prior        - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN       - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond         - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN        - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|~Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)

    [num_class,num_training]=size(train_target);%num_class:Q����ǩ����   num_training��M��������

%Computing distance between training instances
    dist_matrix=diag(realmax*ones(1,num_training));
    for i=1:num_training-1
        if(mod(i,100)==0)
            disp(strcat('computing distance for instance:',num2str(i)));
        end
        vector1=train_data(i,:);
        for j=i+1:num_training            
            vector2=train_data(j,:);
            if (para_p==1)
                dist_matrix(i,j)=sum(abs(vector1-vector2));         %�����پ���
            elseif(para_p==2)
                dist_matrix(i,j)=sqrt(sum((vector1-vector2).^2));   %ŷ����þ���
            else
                dist_matrix(i,j)=max(abs(vector1-vector2));         %�б�ѩ����� 
            end
            dist_matrix(j,i)=dist_matrix(i,j);
        end
    end
    
%Computing Prior and PriorN,����������ʣ��õ� Prior - A Qx1 array      PriorN- A Qx1 array
    for i=1:num_class%num_class:Q����ǩ����   num_training��M��������   train_target - A QxM array,
        temp_Ci=sum(train_target(i,:)==ones(1,num_training));
        Prior(i,1)=(Smooth+temp_Ci)/(Smooth*2+num_training);
        PriorN(i,1)=1-Prior(i,1);
    end

%Computing Cond and CondN,����������     num_class:Q����ǩ����      num_training��M��������
%��ѵ�����ݼ��У���ÿһ������������ڵı�ǩ���������������һ��Q*M��NeighborsTarget����
    NeighborsTarget=[];
    Neighbors=cell(num_training,1); %Neighbors{i,1} stores the Num neighbors of the ith training instance,��cell��������Ԫ�����飬����������Ϊ��Ԫ����
    for i=1:num_training
        [temp,index]=sort(dist_matrix(i,:));
        Neighbors{i,1}=index(1:Num);
        NeighborsTarget=[NeighborsTarget,train_target(:,Neighbors{i,1}(1))];%��������������ı�ǩ������ע���Ӧ��ϵ
    end  %�����Ǽ���ÿ��ʵ����Num�������ʵ���������浽Neighbors��
    
 %���¼�������ÿ�������ı�ǩ��������������ı�ǩ�������ƶȣ���HammingLoss�Լ�RankingLoss�����ۣ����������ֵ�þ��󷵻�
 %train_target   -- NeighborsTarget
    [a,b]=size(NeighborsTarget);
    disp(strcat('NeighborsTarget:',num2str(a),'---:',num2str(b)));
    HammingLoss=Hamming_loss(train_target,NeighborsTarget);
    RankingLoss=Ranking_loss(train_target,NeighborsTarget);
    Average_Precision=Average_precision(train_target,NeighborsTarget);
    LossValue=[HammingLoss,RankingLoss,Average_Precision];
 
    %temp_Ci��Qx(Num+1) array   temp_NCi��Qx(Num+1) array
    temp_Ci=zeros(num_class,Num+1); %The number of instances belong to the ith class which have k nearest neighbors in Ci is stored in temp_Ci(i,k+1)
    temp_NCi=zeros(num_class,Num+1); %The number of instances not belong to the ith class which have k nearest neighbors in Ci is stored in temp_NCi(i,k+1)
  
    for i=1:num_training%ͳ�Ƹ���
        temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
        neighbor_labels=[];
        for j=1:Num%����ÿ��ʵ����Num�����ڵı�ǩ����  neighbor_labels��Q*Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];% train_target - A QxM array
        end
        for j=1:num_class%��ÿ��ʵ����ͳ����Num������ڵı�ǩ������Ӧ��ǩ��ѡ�еĸ�������ֵ��Ϊ��һ�������λ�ò���
            temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));%Num������ڵı�ǩ���У���temp(1,j)����ǩ��Ϊ1
        end
        for j=1:num_class  %temp_Ci��Qx(Num+1) array   temp_NCi��Qx(Num+1) array
            if(train_target(j,i)==1)%train_target - A QxM array,%ʵ���еı�ǩ��ѡ��
                temp_Ci(j,temp(j)+1)=temp_Ci(j,temp(j)+1)+1;%�ڵ�temp(1,j)��λ�ã���ֵ�Լ�1
            else%ʵ���еı�ǩû�б�ѡ��
                temp_NCi(j,temp(j)+1)=temp_NCi(j,temp(j)+1)+1;%�ڵ�temp(1,j)��λ�ã���ֵ�Լ�1
            end
        end
    end
    
    for i=1:num_class %num_class:Q����ǩ����
        temp1=sum(temp_Ci(i,:));%���ÿ����ǩ��num_training��ʵ���У��ñ�ǩ��ѡ�е�ʵ������
        temp2=sum(temp_NCi(i,:));%���ÿ����ǩ��num_training��ʵ���У��ñ�ǩû��ѡ�е�ʵ������
        for j=1:Num+1 %  temp1 + temp2 == num_training
            Cond(i,j)=(Smooth+temp_Ci(i,j))/(Smooth*(Num+1)+temp1);%�������ֵ
            CondN(i,j)=(Smooth+temp_NCi(i,j))/(Smooth*(Num+1)+temp2);%�������ֵ
        end
    end              