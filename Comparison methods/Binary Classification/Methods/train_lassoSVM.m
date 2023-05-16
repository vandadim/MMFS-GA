function [accM,Param,select_ALL,label_P]=train_lassoSVM(X,y,Xtest,ytest)


kfold = 10;
rho_t = [0.0001,0.001,0.01,0.1,0.2,0.4,0.6,0.8,0.9,1];
acc_record = zeros(kfold,length(rho_t));


for i = 1:length(rho_t)
    Indices = crossvalind('Kfold', y, 10);
    rho=rho_t(i);
    for j = 1:kfold
        idx_tr = Indices ~= j;
        idx_te = Indices == j;
        X_tr = [X{1}(idx_tr,:),X{2}(idx_tr,:),X{3}(idx_tr,:),X{4}(idx_tr,:),X{5}(idx_tr,:)];
        y_tr = y(idx_tr);
        X_ts = [X{1}(idx_te,:),X{2}(idx_te,:),X{3}(idx_te,:),X{4}(idx_te,:),X{5}(idx_te,:)];
        y_ts = y(idx_te);
               
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
        [w, funVal1, ValueL1]= LeastR(X_tr, y_tr, rho, opts);
        
        %------------------------- Feature Selection -----------------------
        idx_fea = sum(w.^2,2) > 1e-5;
        X_tr = X_tr(:,idx_fea);
        X_ts = X_ts(:,idx_fea);
        
        %%
        %---------------------------- Run SVM ------------------------------
        options = ['-c 1'];
        model = svmtrain(y_tr,X_tr,options);
        [label, accuracy, dec_values] = svmpredict(y_ts, X_ts,model);
        acc = accuracy(1);
        
        %-------------------------- Save results ---------------------------
        acc_record(j,i) = acc;
       
    end
end
% [acc_max,idx] = max(acc_record(:));
% [j_best,i_best,q_best] = ind2sub(size(acc_record),idx);
Mean_acc=mean(acc_record);
[acc_max,idx] = max(Mean_acc(:));

rho=rho_t(idx);
X_train = [X{1},X{2},X{3},X{4},X{5}];
y_train = y;
X_test = [Xtest{1},Xtest{2},Xtest{3},Xtest{4},Xtest{5}];
[w, funVal1, ValueL1]= LeastR(X_train, y_train, rho, opts);

%------------------------- Feature Selection -----------------------
idx_fea = sum(w.^2,2) > 1e-5;
X_tr = X_train(:,idx_fea);
X_ts = X_test(:,idx_fea);

%%
%---------------------------- Run SVM ------------------------------
options = ['-c 1'];
model = svmtrain(y,X_tr,options);
[label_P, accuracy, dec_values] = svmpredict(ytest, X_ts,model);
accM = accuracy(1);
Param.rho=rho;
select_ALL=find(idx_fea==1);

end