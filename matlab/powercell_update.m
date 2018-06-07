function [ X, w ] = powercell_update( X, w, mu, options )
%POWERCELL_UPDATE Update point positions by progressive refinement.
%   The cost function being optimized can be written out explicitly for a 
%   single mu:
%       F[X, w] = 1/n*\sum_i w_i + \sum_i \int_{V_i} |x_i - x|^2 - w_i d\mu
%
%   By setting gradients wrt w_i and x_i to 0, we recover a simple
%   algorithm based on a gradient ascent solve for the w_i and a fixed
%   point iteration for the x_i. 
%
%   See: https://arxiv.org/abs/1802.05757.pdf
sampler = options.sampler;

n = size(X,1);
m = length(mu);

%% Sample new point and update weights
y = sampler(1);
if n==0
    X(1,:) = y;
    w = zeros(1,m);
else
    X(n+1,:) = y;
    w(n+1,:) = 0;
end

%% Run this for T iterations to ensure convergence (T = 10 works)
for t=1:10
    
%% Update weights
% Solve gradient ascent with \nabla_{w_i} F_j = 1/n - m_i^j
%       m_i^j = \int_{V_i} d\mu_j
for k=1:m
    w(:,k) = weight_update(X,w(:,k),mu{k});
end

%% Move points
% Solved fixed point iteration with \nabla_{x_i} F_j = 2 m_i^j (x_i - b_i^j)
%       m_i^j = \int_{V_i} d\mu_j
%       b_i^j = \int_{V_i} x d\mu_j(x)
n = size(X,1);
Xnew = zeros(size(X));
Msum = zeros(n,1);
for k=1:length(mu)
    M = sample_powercell_density(X,w(:,k),mu{k});
    B = sample_powercell_means(X,w(:,k),mu{k});
    for i=1:n
        B(i,:) = M(i)*B(i,:);
    end
    Xnew = Xnew+B;
    Msum = Msum+M;
end
for i = 1:n
    X(i,:) = Xnew(i,:)/Msum(i);
end

end


