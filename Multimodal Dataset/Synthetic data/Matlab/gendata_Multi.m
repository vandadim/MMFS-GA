function data = gendata_Multi(num_data,options)

% This function generates synthetic multi-view data for classification
% tasks.
% Inputs:
% - num_data: a vector containing the number of data points for each class
% - options: a structure containing optional parameters for data generation
%
% Outputs:
% - data: a structure containing generated data

%% Input parameters
if nargin < 2
    options = []; 
end
options.num_data = num_data;
if ~isfield(options,'noise_features')
    options.noise_features = 20; 
end
if ~isfield(options,'num_classes')
    options.num_classes = 2;
end
if ~isfield(options,'labels')
    options.labels = [1:(options.num_classes)]; 
end

if (length(options.num_data) == 1)
  options.num_data = repmat(options.num_data, 1, options.num_classes);
end

num_data = options.num_data;

% Check if the sum of number of data points is within range
if (sum(num_data) > 65535)
    error('GENSYN: uint16 too short.\n'); 
end

%%
% Assign labels to data points
y = repmat(options.labels(1), 1, options.num_data(1));
for i = 2:options.num_classes
    y = [y repmat(options.labels(i), 1, options.num_data(i))];
end

% Map labels to binary values if num_classes = 2
if options.num_classes == 2
    y(y==2)=0;
end

% Generate covariance matrices and means for informative features of view A
% and view B.
if options.num_classes == 2
    % informative features(View A)    
    C(1:3,1:3,1) = eye(3);    
    C(4:6,4:6,1) = [ 1.05  0.48  0.95; ...
                     0.48   1     0.2; ...
                     0.95  0.2   1.05 ];
    mu(1:3,1)    = 1.1*ones(3,1)/sqrt(3);
    mu(4:6,1)    = [ 0.5 0.4 0 ]';
    
    % informative features(View B)    
    C(7:13,7:13,1) = eye(7);
    mu(7:14,1)     = linspace(1.8,0,8)/sqrt(8);
else
    % informative features(View A)    
    C(1:3,1:3,1) = eye(3);    
    C(4:6,4:6,1) = [ 1.05  0.48  0.95; ...
                     0.48   1     0.2; ...
                     0.95  0.2   1.05 ];
    mu(1:3,1)    = 1.1*ones(3,1)/sqrt(3);
    mu(4:6,1)    = [ 0.5 0.4 0 ]';
    
    % informative features(View B)    
    C(7:10,7:10,1) = eye(4);
    C(11:13,11:13,1) = [ 1.05  0.48  0.95; ...
                         0.48   1     0.2; ...
                         0.95  0.2   1.05 ];
    mu(7:14,1)     = linspace(1.8,0,8)/sqrt(8);
end

% Independent + decreasing power of features
d = options.noise_features;
C(14:(13 + d),14:(13 + d),1) = eye(d);
mu(14:(13 + d),1) = zeros(d,1);

% randomly permute features
idx = [1 13 17 16 2 4 6 3 9 7 20 19 18 5 14 11 15 12 10 8 (21):(13 +d)];
idx1= [1,5,8,6,14,7,10,20,9,19,16,18,2,15,17,4,3,13,12,11 (21):(13 +d)];
C(:,:,1) = C(idx,idx,1);
mu = mu(idx);    

if options.num_classes == 2
    % common covariance matrix
    C(:,:,2) = C;
    mu(:,2)  = -mu;
else
    C(:,:,2) = C;
    mu(:,2)  = -mu;
    for i = 3:options.num_classes
      mu(:,i) = (2*(i - 2) + 1)*mu(:,1);
    end
end
% Generates data from the first class using the mean and covariance matrix.
X = nrand(mu(:,1),C(:,:,1),num_data(1));

% Loops over the remaining classes for generating data.
for i = 2:options.num_classes 
      X = [X nrand(mu(:,i),C(:,:,1),num_data(i))];
end
X=X';
all_data=X(:,idx1);

input_X_Real_1  = all_data(:,1:6);
input_X_Real_2  = all_data(:,7:13);
input_X_Real_both= all_data(:,1:13); 



% View_A
view_A  = all_data(:,14:513);
idx_A   = [1 2 4 6 13 16];
real_fe_A = zeros(size(view_A,2),1);
real_fe_A(idx_A)=1;
view_A(:,idx_A)= input_X_Real_1;
view_A = [view_A y'];
% View_B
view_B  = all_data(:,514:1013);
idx_B   = [1 3 5 7 9 14 17];
real_fe_B = zeros(size(view_B,2),1);
real_fe_B(idx_B)=1;
view_B(:,idx_B)= input_X_Real_2;
view_B = [view_B y'];

% save info
data = struct( ...    
    'x_Real_A',view_A, ...
    'x_Real_B',view_B, ...
    'input_X_Real_A',input_X_Real_1, ...
    'input_X_Real_B',input_X_Real_2, ...
    'input_X_Real_ALL',input_X_Real_both, ...
    'real_A',real_fe_A, ...
    'real_B',real_fe_B, ...
    'y',y, ...
    'dim',size(view_A,1), ...
    'num_classes',options.num_classes, ...
    'num_data',options.num_data, ...
    'labels',options.labels);


