function [threshold] = adaThresh(Image)
%Intuition:
%(1)pixels are divided into two groups
%(2)pixels within each group are very similar to each other
%   Parameters:
%   t : threshold
%   r : pixel value ranging from 1 to 255
%   q_L, q_H : the number of lower and higher group respectively
%   sigma : group variance
%   miu : group mean

nbins = 256;
counts = imhist(Image, nbins);
p = counts / sum(counts);

sigma = zeros(1, nbins);
for t = 1:nbins
    w_L = sum(p(1:t));
    w_H = sum(p(t + 1:end));
    miu_L = sum(p(1:t) .* (1:t)') / w_L;
    miu_H = sum(p(t + 1:end) .* (t + 1:nbins)') / w_H;
    %     sigma(t) = w_L * w_H * (miu_L - miu_H)^2;
    sigma(t) = w_L * miu_L^2 + w_H * miu_H^2; % a proxy for sigma_b
end

[~, threshold] = max(sigma(:));

% new_sigma = p' .* sigma;
% [~, threshold] = max(new_sigma(:));
end
