function [acc,Param,select_ALL,label_P]=train_AMTFS(X,y,Xtest,ytest)

%% Options

opt.repeat = 1:10;
opt.lambda = [0.001, 0.01, 0.1, 1, 5, 10, 30, 40, 60, 100];
opt.mu = [0.001, 0.01, 0.1, 1, 5, 10, 30, 40, 60, 100];
opt.k = 1:1:10;


%%


kfold = 10;
X_tr = cell(5,1);


Indices = crossvalind('Kfold', y, 10);
for j = 1:kfold
    
    % prepare data for training and testing
    idx_tr = Indices ~= j;
    idx_te = Indices == j;
    X_tr{1} = X{1}(idx_tr,:);
    X_tr{2} = X{2}(idx_tr,:);
    X_tr{3} = X{3}(idx_tr,:);
    X_tr{4} = X{4}(idx_tr,:);
    X_tr{5} = X{5}(idx_tr,:);
    X_ts{1} = X{1}(idx_te,:);
    X_ts{2} = X{2}(idx_te,:);
    X_ts{3} = X{3}(idx_te,:);
    X_ts{4} = X{4}(idx_te,:);
    X_ts{5} = X{5}(idx_te,:);
    y_tr = y(idx_tr);
    y_ts = y(idx_te);
    Y = [y_tr,y_tr,y_tr,y_tr,y_tr];
    acc_record = zeros(length(opt.k),length(opt.lambda),length(opt.mu));
    for k = 1:length(opt.k)
        for p = 1:length(opt.lambda)
            for q = 1:length(opt.mu)
                W = AMTFS(X_tr,Y,opt.k(k),opt.lambda(p),opt.mu(q));
                idx_fea = sum(W.^2,2) > 1e-5;
                fea_num = sum(idx_fea);
                
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
                    
                    [acc,label,dec_values] = mkl(Xtr,y_tr,Xts,y_ts,'MTFS');
                    acc_record(k,p,q) = acc;
                end
            end
        end
    end
    MU = mean(acc_record,[1 2]);
    [acc_max_mu,idx_mu] = max(MU(:));
    best_mu_fold(j) = idx_mu;
    best_acc_fold(:,:,j)=acc_record(:,:,idx_mu);
end
[best_mu,id] = mode(best_mu_fold);
best_acc_mu  = best_acc_fold(:,:,id);
[acc_max,idx] = max(best_acc_mu(:));
[k_best,p_best] = ind2sub(size(acc_record),idx);
Y = [y,y,y,y,y];
[W,output] = AMTFS(X,Y,opt.k(k_best),opt.lambda(p_best),opt.mu(best_mu));
Lambda_best=opt.lambda(p_best);
K_best=opt.k(k_best);
Mu_best=opt.mu(best_mu);
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
[acc,label_P,dec_values] = mkl(X_train,y_train,X_test,ytest,'AMTFS');

Param.Lambda_best=Lambda_best;
Param.K_best=K_best;
Param.Mu_best=Mu_best;
select_ALL{1,1}=find(idx1_fea==1);
select_ALL{2,1}=find(idx2_fea==1);
select_ALL{3,1}=find(idx3_fea==1);
select_ALL{4,1}=find(idx4_fea==1);
select_ALL{5,1}=find(idx5_fea==1);
end

