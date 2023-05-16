function [acc_max,label_p,dec_values_p] = mkl(X,y,Xtest,ytest,method)
% compute linear kernel
dataset='ADvsNC';
% method='AMTFS';
modal = length(X);
K = cell(modal,1);
Ktest = cell(modal,1);
% for i = 1:modal
%     K{i} = X{i};% Chnaged from X{i}*X{i}'; to X{i}
%     Ktest{i} = Xtest{i};% Chnaged from Xtest{i}*X{i}'; to Xtest{i}
% end

for i = 1:modal
    K{i} = X{i}*X{i}';
    Ktest{i} = Xtest{i}*X{i}';
end


C = 1;
% w1 = 1;
% w2 = 2;

acc_max = 0;

beta = 0.20;


%                 options = ['-t 4 -c ', num2str(C(j)),' -w1 1 -w-1 10'];
options = ['-t 4 -c ', num2str(C(1))];
% model = svmtrain(y,[ones(size(K{1},1),1),beta*K{1}+beta*K{2}+beta*K{3}+beta*K{4}+beta*K{5}], options);
model = svmtrain(y,[(1:length(y))',beta*K{1}+beta*K{2}+beta*K{3}+beta*K{4}+beta*K{5}], options);
% [label, acc, dec_values] = svmpredict(ytest, [ones(size(Ktest{1},1),1),beta*Ktest{1}+beta*Ktest{2}+beta*Ktest{3}+beta*Ktest{4}+beta*Ktest{5}],model);
[label, acc, dec_values] = svmpredict(ytest, [(1:length(ytest))',beta*Ktest{1}+beta*Ktest{2}+beta*Ktest{3}+beta*Ktest{4}+beta*Ktest{5}],model);
Bal_Acc_Test_Real              = Acc_F1_Balance (ytest, label, 2)
acc_max = Bal_Acc_Test_Real;
if acc_max < acc(1)
    acc_max = Bal_Acc_Test_Real;
    label_p = label;
    dec_values_p = dec_values;
end
    
    
end

