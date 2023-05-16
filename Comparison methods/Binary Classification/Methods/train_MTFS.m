function [accM,Param,select_ALL,label_P]=train_MTFS(X,y,Xtest,ytest)

kfold = 10;
lambda = [0.001,0.01,0.1,0.3,0.5,0.7,0.8,0.9,1,5,10,30,40,60,100];
acc_record = zeros(kfold,length(lambda));


for i = 1:length(lambda)
    Indices = crossvalind('Kfold', y, 10);
    lambda_1=lambda(i);
    for j = 1:kfold
        idx_tr = Indices ~= j;
        idx_te = Indices == j;
        X_tr{1} = X{1}(idx_tr,:);
        X_tr{2} = X{2}(idx_tr,:);
        X_tr{3} = X{3}(idx_tr,:);
        X_tr{4} = X{4}(idx_tr,:);
        X_tr{5} = X{5}(idx_tr,:);
        y_t = y(idx_tr);
        y_tr{1}=y_t;
        y_tr{2}=y_t;
        y_tr{3}=y_t;
        y_tr{4}=y_t;
        y_tr{5}=y_t;
        X_ts{1} = X{1}(idx_te,:);
        X_ts{2} = X{2}(idx_te,:);
        X_ts{3} = X{3}(idx_te,:);
        X_ts{4} = X{4}(idx_te,:);
        X_ts{5} = X{5}(idx_te,:);
        y_ts = y(idx_te);
        
        %===================== Feature selection ==========================
        %%
        
        opts.init = 0;      % guess start point from data.
        opts.tFlag = 1;     % terminate after relative objective value does not changes much.
        opts.tol = 10^-5;   % tolerance.
        opts.maxIter = 1000; % maximum iteration number of optimization.
        
        %---------------------- Multi-task feature selection --------------
        [W funcVal] = Least_L21(X_tr, y_tr, lambda_1, opts);
        
%         idx_fea = sum(W.^2,2) > 1e-5;
        idx1_fea = sum(W(:,1).^2,2) > 1e-5;
        idx2_fea = sum(W(:,2).^2,2) > 1e-5;
        idx3_fea = sum(W(:,3).^2,2) > 1e-5;
        idx4_fea = sum(W(:,4).^2,2) > 1e-5;
        idx5_fea = sum(W(:,5).^2,2) > 1e-5;
        Xtr{1} = X_tr{1}(:,idx1_fea);
        Xtr{2} = X_tr{2}(:,idx2_fea);
        Xtr{3} = X_tr{3}(:,idx3_fea);
        Xtr{4} = X_tr{4}(:,idx4_fea);
        Xtr{5} = X_tr{5}(:,idx5_fea);
        
        Xts{1} = X_ts{1}(:,idx1_fea);
        Xts{2} = X_ts{2}(:,idx2_fea);
        Xts{3} = X_ts{3}(:,idx3_fea);
        Xts{4} = X_ts{4}(:,idx4_fea);
        Xts{5} = X_ts{5}(:,idx5_fea);
        
        [acc,label,dec_values] = mkl(Xtr,y_t,Xts,y_ts,'MTFS');
        acc_record(j,i) = acc;
        
    end
end

Mean_acc=mean(acc_record);
[acc_max,idx] = max(Mean_acc(:));
lambda_1=lambda(idx);
y_tr{1}=y;
y_tr{2}=y;
y_tr{3}=y;
y_tr{4}=y;
y_tr{5}=y;
[W funcVal] = Least_L21(X, y_tr, lambda_1, opts);
idx1_fea = sum(W(:,1).^2,2) > 1e-5;
idx2_fea = sum(W(:,2).^2,2) > 1e-5;
idx3_fea = sum(W(:,3).^2,2) > 1e-5;
idx4_fea = sum(W(:,4).^2,2) > 1e-5;
idx5_fea = sum(W(:,5).^2,2) > 1e-5;


X_train{1} = X{1}(:,idx1_fea);
X_test{1} = Xtest{1}(:,idx1_fea);

X_train{2} = X{2}(:,idx2_fea);
X_test{2} = Xtest{2}(:,idx2_fea);

X_train{3} = X{3}(:,idx3_fea);
X_test{3} = Xtest{3}(:,idx3_fea);

X_train{4} = X{4}(:,idx4_fea);
X_test{4} = Xtest{4}(:,idx4_fea);

X_train{5} = X{5}(:,idx5_fea);
X_test{5} = Xtest{5}(:,idx5_fea);

y_train = y;

[acc,label_P,dec_values] = mkl(X_train,y_train,X_test,ytest,'lassoMKSVM');
accM =acc;
Param.lambda_1=lambda_1;
select_ALL{1,1}=find(idx1_fea==1);
select_ALL{2,1}=find(idx2_fea==1);
select_ALL{3,1}=find(idx3_fea==1);
select_ALL{4,1}=find(idx4_fea==1);
select_ALL{5,1}=find(idx5_fea==1);
end