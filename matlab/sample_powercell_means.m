function [ bary ] = sample_powercell_means( X, w, nu )
%SAMPLE_POWERCELL_MEAN Sample the powercell means of all points
%   The powercell of a point set X with weights w is the set
%   V_i^w = {x | ||x - x_i||^2 - w_i <= ||x - x_j||^2 - w_j, \forall j}
%
%   This fn finds the mean of a Voronoi cell against a given
%   distribution \nu

bary = zeros(size(X));
cnt = 64000;
Y = nu(cnt);

n = size(X,1);
in = zeros(n,1);
if n==1
    bary=mean(Y);
    return;
end

[~,idx] = min(pdist2(X,Y)-w,[],1);
for i=1:cnt
    bary(idx(i),:) = bary(idx(i),:)+Y(i,:);
    in(idx(i)) = in(idx(i))+1;
end
for i=1:n
    bary(i,:) = bary(i,:)/in(i);
end

end

