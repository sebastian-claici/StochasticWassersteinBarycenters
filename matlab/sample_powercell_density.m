function [ rho ] = sample_powercell_density( X, w, nu )
%SAMPLE_POWERCELL_DENSITY Sample the powercell density of V_i
%   The powercell of a point set X with weights w is the set
%   V_i^w = {x | ||x - x_i||^2 - w_i <= ||x - x_j||^2 - w_j, \forall j}
%
%   This fn finds the density of all Voronoi cells against a given
%   distribution \nu

n = size(X, 1);
if n==1
    rho = ones(n,1);
    return;
end

in = zeros(n, 1);
cnt = 64000;
Y = nu(cnt);

[~,idx] = min(pdist2(X,Y)-w,[],1);
for i=1:cnt
    in(idx(i)) = in(idx(i))+1;
end

rho = in / cnt;

end

