function [P] = def_prior(img, sp)

if nargin == 1
    sp = false;
end

%% superpixel?
if sp
    [L, N] = superpixels(img, 500);
    Xsp = zeros(size(img), 'like', img);
    idx = label2idx(L);
    for labelVal = 1:N
        pixIdx = idx{labelVal};
        Xsp(pixIdx) = mean(img(pixIdx));
    end
    img = Xsp;
end

%% extract patches
[H, W, ~] = size(img);
patches = im2patches(img, 32, 16); %[row, col, pH, pW, C]
[row, col, pH, pW, C] = size(patches);
patches = reshape(patches, row*col, pH, pW, C);


%%
feats = zeros(row*col, pH, pW, 8+1);
for Pi = 1:length(patches)
    patch = squeeze(patches(Pi, :, :, :));
    % MR8 filter banks
    patch_pad = padarray(patch, [10,10], 'symmetric');
    mr8_feat = MR8fast(patch_pad);
    mr8_feat = reshape(mr8_feat, 8, pH, pW);
    mr8_feat = permute(mr8_feat, [2,3,1]);
    
    %     % corner extraction
    %     corner_feat = detectHarrisFeatures(patch);
    %     % corner_feat = detectMinEigenFeatures(X);
    
    % edge extraction
    edge_feat = edge(patch, 'sobel');
    % edge_feat = edge(X, 'canny');
    
    feats(Pi, :, :, 1:8) = mr8_feat;
    %     feats(Pi, :, :, end-1) = corner_feat;
    feats(Pi, :, :, end) = edge_feat;
end

%% probability based on similarity
mean_feats = squeeze(mean(feats, [2,3]));
mean_vec = mean(mean_feats, 1);

% P = vecnorm(mean_feats - mean_vec, 2, 2);
P = cosSim(mean_feats, mean_vec);
P = reshape(P, row, col);
P = imresize(P, [H, W], 'nearest');
end