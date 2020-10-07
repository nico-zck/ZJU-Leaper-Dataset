function [g] = gabor_filter(size, s, theta)
assert(size > 1);
size = fix(size/2);
g0 = 1/(2^(2+s/2));
sigma = sqrt(2)/g0;
[X,Y] = meshgrid(-size:size, -size:size);
g = gv(X,Y,theta,g0,sigma);
% g = imresize(g, 0.5);
end

function Gv = gv(X, Y, theta, g0, sigma)
Xp = X.*cosd(theta) + Y.*sind(theta);
Yp = -X.*sind(theta) + Y.*cosd(theta);
% Gv = 1/(2*sigma^2) * exp( -((Xp.^2+Yp.^2) ./ (2*sigma)) ) .* sin(2*pi*g0*Xp);
Gv = exp( -((Xp.^2+Yp.^2) ./ (2*sigma^2)) ) .* sin(2*pi*g0*Xp);
end

