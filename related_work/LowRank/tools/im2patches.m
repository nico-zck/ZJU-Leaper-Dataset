function [patches] = im2patches(img, patch_size, patch_stride)
% IM2PATCHES Extract rectangular patches from input image.
% - I, required, image array
% - patch_size, required
% - patch_stride, optional, default=patch_size
% - return: patches with shape [row, col, pH, pW, C]
%
%   USAGE EXAMPLES:
%       patches = im2patches(I, [pH,pW]);
%       patches = im2patches(I, [pH,pW], [sH,sW]);
%
% Nico Zhang, <ckzhang@zju.edu.cn>
% Last update: October 2020

assert(ismatrix(img) || ndims(img) == 3, 'Input must be a 2D or 3D array');

assert(all(patch_size > 0), 'Patch size cannot be negative or zero')
if isscalar(patch_size)          % Square patches
    pH = patch_size;
    pW = patch_size;
elseif numel(patch_size) == 2    % Rectangular patches
    pH = patch_size(1);
    pW = patch_size(2);
else
    error('patch_ize can be either a scalar or a [pH, pW] vector.')
end

if nargin == 2
    sH = pH;
    sW = pW;
elseif isscalar(patch_stride)
    sH = patch_stride;
    sW = patch_stride;
elseif numel(patch_stride) == 2
    sH = patch_stride(1);
    sW = patch_stride(2);
else
    error('patch_tride can be either a scalar or a [sH, sW] vector.')
end


[H,W,C] = size(img);

row = floor((H-pH)/sH + 1);
col = floor((W-pW)/sW + 1);
patches = zeros(row, col, pH, pW, C);

Ri = 0;
for Hi = [1:sH:H-pH+1; pH:sH:H]
    %     disp([Hi(1), Hi(2)]);
    Ri = Ri + 1;
    Ci = 0;
    for Wi = [1:sW:W-pW+1; pW:sW:W]
        Ci = Ci + 1;
        patches(Ri, Ci, :, :, :) = img(Hi(1):Hi(2), Wi(1):Wi(2), :);
    end
end
% patches = reshape(patches, row, col, pH, pW, C);
end