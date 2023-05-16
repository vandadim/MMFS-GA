function [accM,Param,select_ALL,label_P]=train_M2TFS(X,y,Xtest,ytest)

opt.rho_L3 = [0.001,0.01,0.1,0.3,0.5,0.7,0.8,0.9,1,5,10,30,40,60,100];
opt.rho1 = [0.001,0.01,0.1,0.3,0.5,0.7,0.8,0.9,1,5,10,30,40,60,100];

Ytr{1}=y;
Ytr{2}=y;
Ytr{3}=y;
Ytr{4}=y;
Ytr{5}=y;
kfold = 10

Indices = crossvalind('Kfold', y, 10);
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
    S = f_lapMatrixV2(X_tr,y_tr);
    
    
    opts.init = 0;      % guess start point from data.
    opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    opts.tol = 10^-5;   % tolerance.
    opts.maxIter = 1000; % maximum iteration number of optimization.
    
    acc_record = zeros(length(opt.rho1),length(opt.rho_L3));
    for k = 1:length(opt.rho1)
        k
        for p = 1:length(opt.rho_L3)
            p
            %---------------------- Multi-task feature selection --------------
            opts.rho1 = opt.rho1(k);
            opts.rho_L3 = opt.rho_L3(p);
            
            W = jb_MTM_APG(X_tr,y_tr,opts,S);
            
            idx_fea = sum(W.^2,2) > 1e-5;
            
            if sum(idx_fea)~=0 && sum(idx_fea)~=size(W,1)
                
                
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
                acc_record(k,p) = acc;
            end            
        end
    end
    All_Acc(:,:,j)=acc_record;
end
Mean_all_acc=mean(All_Acc,3);

[acc_max,idx] = max(Mean_all_acc(:));
[k_best,p_best] = ind2sub(size(acc_record),idx);
opts.rho1 = opt.rho1(k_best);
opts.rho_L3 = opt.rho_L3(p_best);
S = f_lapMatrixV2(X,Ytr);
W = jb_MTM_APG(X,Ytr,opts,S);
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
% classification
[acc,label_P,dec_values] = mkl(X_train,y_train,X_test,ytest,'M2TFS');
accM=acc;
Param.Roh1=opts.rho1;
Param.rho_L3=opts.rho_L3;

select_ALL{1,1}=find(idx1_fea==1);
select_ALL{2,1}=find(idx2_fea==1);
select_ALL{3,1}=find(idx3_fea==1);
select_ALL{4,1}=find(idx4_fea==1);
select_ALL{5,1}=find(idx5_fea==1);
end

