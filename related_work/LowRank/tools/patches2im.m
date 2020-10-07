function [image] = patches2im(patches, patch_stride)
% PATCHES2IM Stich together patches extracted from an image
% - patches, required, should with shape [row, col, H, W, C]
% - patch_stride, optional, default=[H,W]
% - return: I, image array
%
% Nico Zhang, <ckzhang@zju.edu.cn>
% Last update: October 2020

[row, col, pH, pW, C] = size(patches);

if nargin == 1
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

Hout = row * sH + (pH - sH);
Wout = col * sW + (pW - sW);
image = zeros(Hout, Wout, C);
image_div = zeros(Hout, Wout);

Ri = 0;
for Hi = [1:sH:Hout-pH+1; pH:sH:Hout]
    %     disp([Hi(1), Hi(2)]);
    Ri = Ri + 1;
    Ci = 0;
    for Wi = [1:sW:Wout-pW+1; pW:sW:Wout]
        Ci = Ci + 1;
        image(Hi(1):Hi(2),Wi(1):Wi(2),:) = image(Hi(1):Hi(2),Wi(1):Wi(2),:) + squeeze(patches(Ri,Ci,:,:,:));
        image_div(Hi(1):Hi(2),Wi(1):Wi(2),:) = image_div(Hi(1):Hi(2),Wi(1):Wi(2),:) + 1;
    end
end

image = image ./ image_div;
end