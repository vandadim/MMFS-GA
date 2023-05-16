%% FUNCTION Least_Lasso
% Sparse Structure-Regularized Learning with Least Squares Loss.
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * \|W\|_1 + opts.rho_L2 * \|W\|_F^2}
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: sprasity controlling parameter
%   opts.rho_L2: L2-norm regularization parameter
%
%% OUTPUT
%   W: model: d * t
%   funcVal: function value vector.
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% Related papers
%
%   [1] Tibshirani, J. Regression shrinkage and selection via
%   the Lasso, Journal of the Royal Statistical Society. Series B 1996
%
%% Related functions
%   Logistic_Lasso, init_opts

%% Code starts here
function [W, funcVal] = IDMTFS(X, Y, rho1, rho2, s, opts)

if nargin <3
    error('\n Inputs: X, Y, abd rho1 should be specified!\n');
end
X = multi_transpose(X);

if nargin <4
    opts = [];
end

% initialize options.
opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

task_num  = length (X);
dimension = size(X{1}, 1);
funcVal = [];

XY = cell(task_num, 1);
W0_prep = [];
for t_idx = 1: task_num
    XY{t_idx} = X{t_idx}*Y{t_idx};
    W0_prep = cat(2, W0_prep, XY{t_idx});
end

% initialize a starting point
if opts.init==2
    W0 = zeros(dimension, task_num);
elseif opts.init == 0
    W0 = W0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0=W0_prep;
    end
end


bFlag=0; % this flag tests whether the gradient step only changes a little

Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;


iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval  (Ws);
    
    while true
        [Wzp l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho1 / gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + trace(delta_Wzp' * gWs) + gamma/2 * norm(delta_Wzp, 'fro')^2;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + rho1 * l1c_wzp);
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;


% private functions

    function [z l1_comp_val] = l1_projection (v, beta)
        % this projection calculates
        % argmin_z = \|z-v\|_2^2 + beta \|z\|_1
        % z: solution
        % l1_comp_val: value of l1 component (\|z\|_1)
        
        z = zeros(size(v));
        vp = v - beta/2;
        z (v> beta/2)  = vp(v> beta/2);
        vn = v + beta/2;
        z (v< -beta/2) = vn(v< -beta/2);
        
        
        l1_comp_val = sum(sum(abs(z)));
    end

    function [grad_W] = gradVal_eval(W)
        if opts.pFlag
            grad_W = zeros(size(W));
            parfor t_ii = 1:task_num
                XWi = X{t_ii}' * W(:,t_ii);
                XTXWi = X{t_ii}* XWi;
                grad_W(:,t_ii) = XTXWi - XY{t_ii};
            end
        else
            grad_W = [];
            for t_ii = 1:task_num
                XWi = X{t_ii}' * W(:,t_ii);
                XTXWi = X{t_ii}* XWi;
                grad_W = cat(2, grad_W, XTXWi - XY{t_ii});
            end
        end
        grad_W(:,1) = grad_W(:,1) + rho2*(2*X{1}*diag(s)*X{1}'*W(:,1) - 2*X{1}*diag(s)*X{2}'*W(:,2));
        grad_W(:,2) = grad_W(:,2) + rho2*(2*X{2}*diag(s)*X{2}'*W(:,2) - 2*X{2}*diag(s)*X{1}'*W(:,1));
        grad_W = grad_W + rho_L2 * 2 * W;
    end

    function [funcVal] = funVal_eval (W)
        
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        else
            for i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        end
        funcVal = funcVal + ...
            rho2*(W(:,1)'*X{1}*diag(s)*X{1}'*W(:,1) + W(:,2)'*X{2}*diag(s)*X{2}'*W(:,2) - ...
    2*W(:,1)'*X{1}*diag(s)*X{2}'*W(:,2));
        funcVal = funcVal + rho_L2 * norm(W, 'fro')^2;
    end

end