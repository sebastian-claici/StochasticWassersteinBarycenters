clear; clc;

mu{1} = @(n) [4 4]+randn(n,2);
mu{2} = @(n) [-4 -4]+randn(n,2);
options.sampler = @(n) [-8 -8]+16*rand(n,2);

%% Initial normal distribution
x1 = -7:.2:7;
x2 = -7:.2:7;
[X1, X2] = meshgrid(x1, x2);

F{1} = mvnpdf([X1(:) X2(:)], [4 4], eye(2));
F{1} = reshape(F{1}, length(x2), length(x1));
F{2} = mvnpdf([X1(:) X2(:)], [-4 -4], eye(2));
F{2} = reshape(F{2}, length(x2), length(x1));

%% Iteratively show each barycenter point as it is being added
filename = 'fw-updates.gif';
fig = figure;

hold on;
% h1 = fill([-5 -5 -3 -3 -5], [-5 -3 -3 -5 -5], 'r');
% h2 = fill([3 3 5 5 3], [3 5 5 3 3], 'b');
% set(h1, 'facealpha', .3);
% set(h2, 'facealpha', .3);
contour(x1, x2, 5 * F{1}, [.001 .01 .05:.1:.95 .99 .999 .9999]); 
contour(x1, x2, 5 * F{2}, [.001 .01 .05:.1:.95 .99 .999 .9999]);
axis([-8 8 -8 8]);

%grid on;
axis on;
axis equal;

iters = 100;
X = [];
w = [];
for i=1:iters
    fprintf('Iteration %d\n', i);

    [X, w] = powercell_update(X, w, mu, options);
    s = scatter(X(:,1), X(:,2), 50, [0.9 0.9 0],...
                'filled', 'MarkerEdgeColor', [0 0 0],...
                'LineWidth', 1);

    frame = getframe(fig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);

    if i==1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.5);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.5);
    end

    if i~=100
        delete(s);
    end
end