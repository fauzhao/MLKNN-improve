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

    [num_class,num_training]=size(train_target);%num_class:Q，标签个数   num_training：M，样本数

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
                dist_matrix(i,j)=sum(abs(vector1-vector2));         %曼哈顿距离
            elseif(para_p==2)
                dist_matrix(i,j)=sqrt(sum((vector1-vector2).^2));   %欧几里得距离
            else
                dist_matrix(i,j)=max(abs(vector1-vector2));         %切比雪夫距离 
            end
            dist_matrix(j,i)=dist_matrix(i,j);
        end
    end
    
%Computing Prior and PriorN,计算先验概率，得到 Prior - A Qx1 array      PriorN- A Qx1 array
    for i=1:num_class%num_class:Q，标签个数   num_training：M，样本数   train_target - A QxM array,
        temp_Ci=sum(train_target(i,:)==ones(1,num_training));
        Prior(i,1)=(Smooth+temp_Ci)/(Smooth*2+num_training);
        PriorN(i,1)=1-Prior(i,1);
    end

%Computing Cond and CondN,计算后验概率     num_class:Q，标签个数      num_training：M，样本数
%在训练数据集中，将每一个样本的最近邻的标签向量组合起来生成一个Q*M的NeighborsTarget矩阵
    NeighborsTarget=[];
    Neighbors=cell(num_training,1); %Neighbors{i,1} stores the Num neighbors of the ith training instance,用cell函数创建元胞数组，创建的数组为空元胞。
    for i=1:num_training
        [temp,index]=sort(dist_matrix(i,:));
        Neighbors{i,1}=index(1:Num);
        NeighborsTarget=[NeighborsTarget,train_target(:,Neighbors{i,1}(1))];%生成最近邻样本的标签集矩阵，注意对应关系
    end  %以上是计算每个实例的Num个最近邻实例，并储存到Neighbors中
    
 %以下几步计算每个样本的标签集与最近邻样本的标签集的相似度，用HammingLoss以及RankingLoss来评价，并将计算的值用矩阵返回
 %train_target   -- NeighborsTarget
    [a,b]=size(NeighborsTarget);
    disp(strcat('NeighborsTarget:',num2str(a),'---:',num2str(b)));
    HammingLoss=Hamming_loss(train_target,NeighborsTarget);
    RankingLoss=Ranking_loss(train_target,NeighborsTarget);
    Average_Precision=Average_precision(train_target,NeighborsTarget);
    LossValue=[HammingLoss,RankingLoss,Average_Precision];
 
    %temp_Ci：Qx(Num+1) array   temp_NCi：Qx(Num+1) array
    temp_Ci=zeros(num_class,Num+1); %The number of instances belong to the ith class which have k nearest neighbors in Ci is stored in temp_Ci(i,k+1)
    temp_NCi=zeros(num_class,Num+1); %The number of instances not belong to the ith class which have k nearest neighbors in Ci is stored in temp_NCi(i,k+1)
  
    for i=1:num_training%统计个数
        temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
        neighbor_labels=[];
        for j=1:Num%给出每个实例的Num个近邻的标签集合  neighbor_labels：Q*Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];% train_target - A QxM array
        end
        for j=1:num_class%对每个实例，统计其Num个最近邻的标签集中相应标签被选中的个数，其值作为下一步计算的位置参数
            temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));%Num个最近邻的标签集中，有temp(1,j)个标签都为1
        end
        for j=1:num_class  %temp_Ci：Qx(Num+1) array   temp_NCi：Qx(Num+1) array
            if(train_target(j,i)==1)%train_target - A QxM array,%实例中的标签被选中
                temp_Ci(j,temp(j)+1)=temp_Ci(j,temp(j)+1)+1;%在第temp(1,j)个位置，该值自加1
            else%实例中的标签没有被选中
                temp_NCi(j,temp(j)+1)=temp_NCi(j,temp(j)+1)+1;%在第temp(1,j)个位置，该值自加1
            end
        end
    end
    
    for i=1:num_class %num_class:Q，标签个数
        temp1=sum(temp_Ci(i,:));%针对每个标签，num_training个实例中，该标签被选中的实例个数
        temp2=sum(temp_NCi(i,:));%针对每个标签，num_training个实例中，该标签没被选中的实例个数
        for j=1:Num+1 %  temp1 + temp2 == num_training
            Cond(i,j)=(Smooth+temp_Ci(i,j))/(Smooth*(Num+1)+temp1);%计算概率值
            CondN(i,j)=(Smooth+temp_NCi(i,j))/(Smooth*(Num+1)+temp2);%计算概率值
        end
    end              