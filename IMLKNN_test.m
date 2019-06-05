function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=IMLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN)
%MLKNN_test tests a multi-label k-nearest neighbor classifier.
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN)
%
%    Description
%
%       KNNML_test takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           Num              - Number of neighbors used in the k-nearest neighbor algorithm
%           Prior            - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN           - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond             - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN            - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|~Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)
%      and returns,
%           HammingLoss      - The hamming loss on testing data
%           RankingLoss      - The ranking loss on testing data
%           OneError         - The one-error on testing data as
%           Coverage         - The coverage on testing data as
%           Average_Precision- The average precision on testing data
%           Outputs          - A QxM2 array, the probability of the ith testing instance belonging to the jCth class is stored in Outputs(j,i)
%           Pre_Labels       - A QxM2 array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1

    [num_class,num_training]=size(train_target);
    [num_class,num_testing]=size(test_target);
    
%Computing distances between training instances and testing instances
    dist_matrix=zeros(num_testing,num_training);%num_testing Ϊ����������ά����ֵΪM2   num_training��M1
    for i=1:num_testing
        if(mod(i,100)==0)
            disp(strcat('computing distance for instance:',num2str(i)));
        end
        vector1=test_data(i,:);
        for j=1:num_training
            vector2=train_data(j,:);
            dist_matrix(i,j)=sqrt(sum((vector1-vector2).^2));%dist_matrix(M2*M1):ÿһ�е�������������һ����������������ѵ�������ľ���ֵ
        end
    end

%Find neighbors of each testing instance
    Neighbors=cell(num_testing,1); %Neighbors{i,1} stores the Num neighbors of the ith testing instance
    for i=1:num_testing%num_testing Ϊ����������ά����ֵΪM2
        [temp,index]=sort(dist_matrix(i,:));%�Ծ���ֵ��С��������
        Neighbors{i,1}=index(1:Num);%��ÿ������������Num������ڵ������Ŵ洢��Neighbors{i,1}��
    end
    
%Computing Outputs
    Outputs=zeros(num_class,num_testing);
    for i=1:num_testing
%         if(mod(i,100)==0)
%             disp(strcat('computing outputs for instance:',num2str(i)));
%         end
        temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
        neighbor_labels=[];
        for j=1:Num %train_target     - A QxM1 array, 
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];
        end
        for j=1:num_class
            temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));
        end
        %disp(sprintf('��%d��ʵ����������ǣ�%d',i,Neighbors{i,1}(1)));
        for j=1:num_class
            %������ MLKNN�㷨(Ӣ��)
            Prob_in=Prior(j)*Cond(j,temp(1,j)+1);%������ܱ�ѡ�еĸ���
            Prob_out=PriorN(j)*CondN(j,temp(1,j)+1);%�������û��ѡ�еĸ���
            
            %2016.01.20:����ʦ�ٴθĽ���1������ڵı�ǩֵ*ϵ��1 + �������ֵ*ϵ��2
            %c1 = 0.05  ; c2 =0.95;
            c1  = 1/Num;   c2 = (Num-1)/Num;
            Prob_in = c1* train_target(j,Neighbors{i,1}(1)) + c2 *Prob_in;
            Prob_out = c1*train_target(j,Neighbors{i,1}(1)) + c2 *Prob_out;
%             
%             if(mod(i,500)==0)
%                 disp(strcat('c1:',num2str(c1),'---c2:',num2str(c2)));
%             end
%              
            if(Prob_in+Prob_out==0)
                Outputs(j,i)=Prior(j);
            else
                Outputs(j,i)=Prob_in/(Prob_in+Prob_out);%�������Ը���ֵ      
            end
        end
    end
    
%Evaluation
    Pre_Labels=zeros(num_class,num_testing);
    for i=1:num_testing
        for j=1:num_class
            if(Outputs(j,i)>=0.5)  %���ڵ���0.5�ű�ʾ��ѡ��
                Pre_Labels(j,i)=1;
            else
                %Pre_Labels(j,i)=-1;%��ǩû��ѡ�У���������Ϊ-1
                Pre_Labels(j,i)=0;%��ǩû��ѡ�У���������Ϊ0
            end
        end
    end
    HammingLoss=Hamming_loss(Pre_Labels,test_target);
    %Accuracy = 1-HammingLoss;
    RankingLoss=Ranking_loss(Outputs,test_target);
    OneError=One_error(Outputs,test_target);
    Coverage=coverage(Outputs,test_target);
    Average_Precision=Average_precision(Outputs,test_target);