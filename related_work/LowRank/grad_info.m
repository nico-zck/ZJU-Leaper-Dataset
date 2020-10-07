function [G] = grad_info(X, alpha)

[Gx, Gy] = imgradientxy(X, 'prewitt');
% M = max(abs(Gx), abs(Gy));
% M = abs(Gx + Gy);
M = sqrt(Gx.^2 + Gy.^2);

% [Gmag, Gdir] = imgradient(X, 'prewitt');
% M = Gmag;

% set threshold for gradient fild
if nargin == 1
alpha = 0.3;
% alpha = -1;
end
L = max(M(:)) - min(M(:));
T = alpha * L;

S = -sign(M - T) .* (M / T);
Phi = abs(M - T) .^ S;
G = 2 * (1 ./ (1 + exp(-Phi)) - 0.5);
end
