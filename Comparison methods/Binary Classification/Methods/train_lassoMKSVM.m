function [accM,Param,select_ALL,label_P]=train_lassoMKSVM(X,y,Xtest,ytest)

kfold = 10;
rho_t = [0.001,0.01,0.1,0.3,0.5,0.7,0.8,0.9,1];
acc_record = zeros(kfold,length(rho_t));


for i = 1:length(rho_t)
    Indices = crossvalind('Kfold', y, 10);
    rho=rho_t(i);
    for j = 1:kfold
        idx_tr = Indices ~= j;
        idx_te = Indices == j;
        X_tr{1} = X{1}(idx_tr,:);
        X_tr{2} = X{2}(idx_tr,:);
        X_tr{3} = X{3}(idx_tr,:);
        X_tr{4} = X{4}(idx_tr,:);
        X_tr{5} = X{5}(idx_tr,:);
        y_tr = y(idx_tr);
        X_ts{1} = X{1}(idx_te,:);
        X_ts{2} = X{2}(idx_te,:);
        X_ts{3} = X{3}(idx_te,:);
        X_ts{4} = X{4}(idx_te,:);
        X_ts{5} = X{5}(idx_te,:);
        y_ts = y(idx_te);
        
        %===================== Feature selection ==========================
        %----------------------- Set optional items ------------------------
        opts=[];
        
        % Starting point
        opts.init=2;        % starting from a zero point
        
        % termination criterion
        opts.tFlag=5;       % run .maxIter iterations
        opts.maxIter=1000;   % maximum number of iterations
        
        % normalization
        opts.nFlag=0;       % without normalization
        
        % regularization
        opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
        
        %----------------------- Run the code LeastR -----------------------
        opts.mFlag=0;       % treating it as compositive function
        opts.lFlag=0;       % Nemirovski's line search
        [w1, funVal1, ValueL1]= LeastR(X_tr{1}, y_tr, rho, opts);
        [w2, funVal1, ValueL1]= LeastR(X_tr{2}, y_tr, rho, opts);
        [w3, funVal1, ValueL1]= LeastR(X_tr{3}, y_tr, rho, opts);
        [w4, funVal1, ValueL1]= LeastR(X_tr{4}, y_tr, rho, opts);
        [w5, funVal1, ValueL1]= LeastR(X_tr{5}, y_tr, rho, opts);
        
        %------------------------- Feature Selection -----------------------
        idx1_fea = sum(w1.^2,2) > 1e-5;
        X_tr{1} = X_tr{1}(:,idx1_fea);
        X_ts{1} = X_ts{1}(:,idx1_fea);
        
        idx2_fea = sum(w2.^2,2) > 1e-5;
        X_tr{2} = X_tr{2}(:,idx2_fea);
        X_ts{2} = X_ts{2}(:,idx2_fea);
        
        idx3_fea = sum(w3.^2,2) > 1e-5;
        X_tr{3} = X_tr{3}(:,idx3_fea);
        X_ts{3} = X_ts{3}(:,idx3_fea);
        
        idx4_fea = sum(w4.^2,2) > 1e-5;
        X_tr{4} = X_tr{4}(:,idx4_fea);
        X_ts{4} = X_ts{4}(:,idx4_fea);
        
        idx5_fea = sum(w5.^2,2) > 1e-5;
        X_tr{5} = X_tr{5}(:,idx5_fea);
        X_ts{5} = X_ts{5}(:,idx5_fea);
        
        %%
        %---------------------- Run multi-kernel SVM ----------------------
        [acc,label,dec_values] = mkl(X_tr,y_tr,X_ts,y_ts,'lassoMKSVM');
        acc_record(j,i) = acc;
        
    end
end
Mean_acc=mean(acc_record);
[acc_max,idx] = max(Mean_acc(:));

rho=rho_t(idx);
[w1, funVal1, ValueL1]= LeastR(X{1}, y, rho, opts);
[w2, funVal1, ValueL1]= LeastR(X{2}, y, rho, opts);
[w3, funVal1, ValueL1]= LeastR(X{3}, y, rho, opts);
[w4, funVal1, ValueL1]= LeastR(X{4}, y, rho, opts);
[w5, funVal1, ValueL1]= LeastR(X{5}, y, rho, opts);
idx1_fea = sum(w1.^2,2) > 1e-5;
X_train{1} = X{1}(:,idx1_fea);
X_test{1} = Xtest{1}(:,idx1_fea);

idx2_fea = sum(w2.^2,2) > 1e-5;
X_train{2} = X{2}(:,idx2_fea);
X_test{2} = Xtest{2}(:,idx2_fea);

idx3_fea = sum(w3.^2,2) > 1e-5;
X_train{3} = X{3}(:,idx3_fea);
X_test{3} = Xtest{3}(:,idx3_fea);

idx4_fea = sum(w4.^2,2) > 1e-5;
X_train{4} = X{4}(:,idx4_fea);
X_test{4} = Xtest{4}(:,idx4_fea);

idx5_fea = sum(w5.^2,2) > 1e-5;
X_train{5} = X{5}(:,idx5_fea);
X_test{5} = Xtest{5}(:,idx5_fea);

y_train = y;

[acc,label_P,dec_values] = mkl(X_train,y_train,X_test,ytest,'lassoMKSVM');
accM =acc;
Param.rho=rho;
%%
select_ALL{1,1}=find(idx1_fea==1);
select_ALL{2,1}=find(idx2_fea==1);
select_ALL{3,1}=find(idx3_fea==1);
select_ALL{4,1}=find(idx4_fea==1);
select_ALL{5,1}=find(idx5_fea==1);
end