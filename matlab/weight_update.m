function [ w ] = weight_update( X, w, mu )
%WEIGHT_UPDATE Update weights using gradient ascent.
%   The ascent step is given by solving for the gradients of
%      F[w] = 1/n * \sum_i w_i + \sum_i \int_{V_i} |x_i - x|^2 - w_i d\mu
%
%   Explicitly, we can compute
%      dF/dw_i = 1/n - \int_{V_i} d\mu
%   which leads to a simple iterative approach
%      w_i^{t+1} = w_i^t + dF/dw_i
%
%   To speed up convergence, we use an accelerated gradients approach.
%
%   See: https://arxiv.org/abs/1802.05757.pdf
n = length(w);
grad = 1/n-sample_powercell_density(X, w, mu);

alpha = 1e-3;
beta = 0.99;
z = zeros(size(w));
normGrad = norm(grad);
iter = 1;
while normGrad > 1e-4
    if mod(iter,100)==0
        fprintf('Iter: %d (norm: %f)\n',iter,normGrad);
    end
    iter = iter+1;

    z = beta*z+grad;
    w = w+alpha*z;
    
    grad = 1/n-sample_powercell_density(X, w, mu);
    normGrad = norm(grad);
end

end

