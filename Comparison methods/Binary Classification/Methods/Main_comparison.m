
%%
% This script loads data from Train and Test sets and performs feature selection
% using different methods (i.e., AMTFS, LASSOsvm, ...) to evaluate their accuracy.
%
% Written by Vandad Imani.
%
% Usage: Parameters are defined in the function and according to a grid
% search it will find the best paratemers.
%
% Inputs: None
%
% Outputs:
% - acc_*: accuracy of * method for each data view (1x10 double)
% - Param_*: model parameters of * method for each data view (1x10 struct)
% - select_*: selected features by * method for each data view (10x1 cell)
% - label_P_*: predicted class labels of * method for each data view (10x1 cell)
% - dec_values_p_*: probabaility class labels of * method for each data view (10x1 cell)
%
% Notes:
% - This code uses external functions 'train_*' to perform feature selection methods.
% - The accuracy is evaluated for each data view separately (total 10 data views).

% Dependencies:
% - train_AMTFS.m
% - train_lassoSVM.m
% - train_IDMTFS.m
% - train_lassoMKSVM.m
% - train_M2TFS.m
% - train_MTFS.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

% Path of train data folder
Data_Folder_train = '..\Train_DATA'; %'D:\MMFSGA_Algorithm\tadpole_data\Data_NC_DE\Train';

% Path of test data folder
Data_Folder_test  = '..\Test_DATA'; %'D:\MMFSGA_Algorithm\tadpole_data\New_Data_NC_MCI_AD\Test';


Data_view_1_Ts        = readtable([Data_Folder_test,'\Data_A_Test.csv']);
Data_view_2_Ts        = readtable([Data_Folder_test,'\Data_Test_B.csv']);
Data_view_3_Ts        = readtable([Data_Folder_test,'\Data_Uniform_Noise.csv']);
Data_view_4_Ts        = readtable([Data_Folder_test,'\Data_ChiSq_Noise.csv']);
Data_view_5_Ts        = readtable([Data_Folder_test,'\Data_Normal_Noise.csv']);
Class_Ts              = Data_view_1_Ts(:,end);

Data_view_1_Ts(:,end)=[];
Data_view_2_Ts(:,end)=[];
Data_view_3_Ts(:,end)=[];
Data_view_4_Ts(:,end)=[];
Data_view_5_Ts(:,end)=[];
Xtest = cell(5,1);
Xtest{1} = Data_view_1_Ts;
Xtest{2} = Data_view_2_Ts;
Xtest{3} = Data_view_3_Ts;
Xtest{4} = Data_view_4_Ts;
Xtest{5} = Data_view_5_Ts;
ytest    = Class_Ts;
ytest(ytest==0)=-1;
for Data_i = 1:10
    Data_view_1        = readtable([Data_Folder_train,'\Data_',num2str(Data_i),'\New_Data_A.csv']);
    Data_view_2        = readtable([Data_Folder_train,'\Data_',num2str(Data_i),'\New_Data_B.csv']);
    Data_view_3        = readtable([Data_Folder_train,'\Data_',num2str(Data_i),'\New_Data_Uniform_Noise.csv']);
    Data_view_4        = readtable([Data_Folder_train,'\Data_',num2str(Data_i),'\New_Data_ChiSq_Noise.csv']);
    Data_view_5        = readtable([Data_Folder_train,'\Data_',num2str(Data_i),'\New_Data_Normal_Noise.csv']);
    
    Data_view_1        = table2array(Data_view_1);
    Class              = Data_view_1(:,end);
    Data_view_1(:,end) = [];
    Data_view_2        = table2array(Data_view_2);
    Data_view_2(:,end) = [];
    Data_view_3        = table2array(Data_view_3);
    Data_view_3(:,end) = [];
    Data_view_4        = table2array(Data_view_4);
    Data_view_4(:,end) = [];
    Data_view_5        = table2array(Data_view_5);
    Data_view_5(:,end) = [];
    
    X{1} = Data_view_1;
    X{2} = Data_view_2;
    X{3} = Data_view_3;
    X{4} = Data_view_4;
    X{5} = Data_view_5;
    y    = Class;
    y(y==0)=-1;
    
    
    %% Feature Selection    
    
	% AMTFS
    [acc_AMTFS(Data_i),Param_AMTFS,select_ALL_AMTF{Data_i,1},label_P_AMTF{Data_i,1}] = train_AMTFS(X,y,Xtest,ytest);
    
	% LASSOsvm
    [acc_LASSOsvm(Data_i),Param_LASSOsvm,select_ALL_LASSOsvm{Data_i,1},label_P_LASSOsvm{Data_i,1}] = train_lassoSVM(X,y,Xtest,ytest);
    
	% LASSOmKsvm
    [acc_LASSOmksvm(Data_i),Param_LASSOmksvm,select_ALL_LASSOmksvm{Data_i,1},label_P_LASSOmksvm{Data_i,1}] = train_lassoMKSVM(X,y,Xtest,ytest);
    
	% train_MTFS
    [acc_MTFS(Data_i),Param_MTFS,select_ALL_MTFS{Data_i,1},label_P_MTFS{Data_i,1}]=train_MTFS(X,y,Xtest,ytest);
    
	% train_M2TFS
    [accM_M2TFS(Data_i),Param_M2TFS,select_ALL_M2TFS{Data_i,1},label_P_M2TFS{Data_i,1}]=train_M2TFS(X,y,Xtest,ytest);
    
    % train_IDMTFS
    [accM_IDMTFS(Data_i),Param_IDMTFS,select_ALL_IDMTFS{Data_i,1},label_P_IDMTFS{Data_i,1}]=train_IDMTFS(X,y,Xtest,ytest);
end

%Save Results
Results.AMTFS.acc=acc_AMTFS;
Results.AMTFS.select_ALL=select_ALL_AMTF;
Results.AMTFS.Label_P=label_P_AMTF;
Results.AMTFS.Param=Param_AMTFS;

Results.LASSOsvm.acc=acc_LASSOsvm;
Results.LASSOsvm.select_ALL=select_ALL_LASSOsvm;
Results.LASSOsvm.Label_P=label_P_LASSOsvm;
Results.LASSOsvm.Param=Param_LASSOsvm;

Results.LASSOmksvm.acc=acc_LASSOmksvm;
Results.LASSOmksvm.select_ALL=select_ALL_LASSOmksvm;
Results.LASSOmksvm.Label_P=label_P_LASSOmksvm;
Results.LASSOmksvm.Param=Param_LASSOmksvm;

Results.MTFS.acc=acc_MTFS;
Results.MTFS.select_ALL=select_ALL_MTFS;
Results.MTFS.Label_P=label_P_MTFS;
Results.MTFS.Param=Param_MTFS;

Results.M2TFS.acc=accM_M2TFS;
Results.M2TFS.select_ALL=select_ALL_M2TFS;
Results.M2TFS.Label_P=label_P_M2TFS;
Results.M2TFS.Param=Param_M2TFS;

Results.IDMTFS.acc=accM_IDMTFS;
Results.IDMTFS.select_ALL=select_ALL_IDMTFS;
Results.IDMTFS.Label_P=label_P_IDMTFS;
Results.IDMTFS.Param=Param_IDMTFS;

save('Results.mat','Results')