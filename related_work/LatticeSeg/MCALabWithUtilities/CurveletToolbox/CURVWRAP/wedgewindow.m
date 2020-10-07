function [wl,wr] = window(x)

% window.m - Creates the two halves of a C^inf compactly supported window
%
% Inputs
%   x       vector or matrix of abscissae, the relevant ones from 0 to 1
%
% Outputs
%   wl,wr   vector or matrix containing samples of the left, resp. right
%           half of the window
%
% Used in FCT.m and FICT.m
%
% Copyright (c) Laurent Demanet, 2004

wr = zeros(size(x));
wl = zeros(size(x));
x(abs(x) < 2^-52) = 0;
wr((x > 0) & (x < 1)) = exp(1-1./(1-exp(1-1./x((x > 0) & (x < 1)))));
wr(x <= 0) = 1;
wl((x > 0) & (x < 1)) = exp(1-1./(1-exp(1-1./(1-x((x > 0) & (x < 1))))));
wl(x >= 1) = 1;
normalization = sqrt(wl.^2 + wr.^2);
wr = wr ./ normalization;
wl = wl ./ normalization;

