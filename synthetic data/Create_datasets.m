clc
clear all
close all
%%
% This script generates 10 multimodal datasets with different combinations of real and
% simulated data and different types of noise, and saves them as CSV files.
% It also saves some details about the real data used in each dataset.
%%

dataset_num = 10;% number of datasets to generate
sample_num = 100;% number of samples in each class

% Options for data generation
datagen_options.num_classes = 2;% number of classes to generate
datagen_options.noise_features = 2000; % number of noise features


% Loop over each dataset to generate and save data
for i=1:dataset_num
    
    % Generate multimodal dataset with noise
    data = gendata_Multi(sample_num,datagen_options);   
    
    % Extract data from different views
    view_A = data.x_Real_A;
    view_B = data.x_Real_B;
    
    % Extract selected features from different views
    input_X_Real_A    = data.input_X_Real_A;
    input_X_Real_B    = data.input_X_Real_B;
    
    % Extract real data from different views
    select_fe_A = data.real_A;
    select_fe_B = data.real_B;    
    
    % Extract labels
    label = data.y;    
    
    % Add noise to different views
    view_Normal_Noise = randn(200,size(select_fe_B,1));
    view_Normal_Noise = [view_Normal_Noise label'];
    % ChiSquareNoise
    view_ChiSq_Noise  = chi2rnd(1,200, size(select_fe_B,1));
    view_ChiSq_Noise = [view_ChiSq_Noise label'];
    % UniformNoise
    view_Uniform_Noise = rand(200,size(select_fe_B,1));
    view_Uniform_Noise = [view_Uniform_Noise label'];
    
    % Create folder to store data
    Folder_name = ['Data_' num2str(i)]; 
    mkdir (Folder_name)
    FullPath    = fullfile(Folder_name);
    
    % Save data to CSV files
    csvwrite([FullPath,'\Data_A.csv'],view_A);
    csvwrite([FullPath,'\Data_B.csv'],view_B);
    csvwrite([FullPath,'\Real_A.csv'],select_fe_A);
    csvwrite([FullPath,'\Real_B.csv'],select_fe_B);
    csvwrite([FullPath,'\Data_Normal_Noise.csv'],view_Normal_Noise);
    csvwrite([FullPath,'\Data_ChiSq_Noise.csv'],view_ChiSq_Noise);
    csvwrite([FullPath,'\Data_Uniform_Noise.csv'],view_Uniform_Noise);
    
    % Create sub-folder to store details about the real data
    Sub_Folder = ['Data_', num2str(i),'\Details'];
    mkdir (Sub_Folder)
    FullPath_Sub    = fullfile(Sub_Folder);
    csvwrite([FullPath_Sub,'\Data_A_Real.csv'],input_X_Real_A);
    csvwrite([FullPath_Sub,'\Data_B_Real.csv'],input_X_Real_B);
    
    % Save details about the real data to a MAT file
    save([FullPath_Sub,'\ALL_DATA_Details.mat'],'data')        
end