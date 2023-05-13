function X=nrand(mu,C,num_data)
% NRAND Generates normally distributed multivariate data.
% 
% Synopsis:
%  X=nrand(mu,C,num_data)
%
% Description:
%  This function randomly generates normally distributed 
%  multivariate data. The normal distrubtion is defined by
%  given covariance matrix C and mean vector mu. 
%
% Input:
%  mu [dim x 1] Mean vectors. 
%  C [dim x dim] Covariance matrix.
%  num_data [int] Number of samples to be generated.
%  
% Output:
%  X [dim x num_data] Generated samples.
%  
% Example:
%  figure; 
%  ppatterns( nrand([1 1]',[2 0;0 1],100) );
%
% See also: 
%  NMIX.
%

% About: Statistical Pattern Recognition Toolbox
% (C) 1999-2003, Written by Vojtech Franc and Vaclav Hlavac
% <a href="http://www.cvut.cz">Czech Technical University Prague</a>
% <a href="http://www.feld.cvut.cz">Faculty of Electrical Engineering</a>
% <a href="http://cmp.felk.cvut.cz">Center for Machine Perception</a>

% Modifications:
%   2-nov-2004, Pavel Krizek, bug in dewhitening transf. fixed, eig replaced by svd
%  17-mar-2004, Pavel Krizek, computation by dewhitening transformation
%  10-jul-2003, VF

% get dimension
dim=size(mu,1);

% compute transf. matrices
%[U,L] = eig(C);
[foo,L,U] = svd(C);

l  = abs(diag(L));

% dewhitening transform
X = U*diag(sqrt(l))*randn(dim,num_data)+repmat(mu,1,num_data);

% computes transf. matrices
%T = inv(chol(inv(C)));
%X = T*randn(dim,num_data)+repmat(mu,1,num_data);

return;
