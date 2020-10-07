function [D, varargout] = LSG_on_img(img, d_star)
% % img = im2double(rgb2gray(imread('bstar_bn_d3.bmp')));
% img = im2double(rgb2gray(imread('bstar_bn_d1.bmp')));
% % img = im2double(rgb2gray(imread('bstar_tk_d3.bmp')));

%% Morphological Component Analysis
[cartoon, texture] = MCA(img); % too slow for now, need to accelerate
% figure; subplot(131); imshow(img, []); subplot(132); imshow(cartoon, []); subplot(133); imshow(texture, []);

%% lattice segmentation
[Lh, Lv, Sh, Sv] = latticeSeg(cartoon);
% figure; imshow(img); hold on;
% for hline = Sh
%     line([hline, hline], [1, 256], 'Color', 'r', 'Linewidth', 1);
% end
% for vline = Sv
%     line([1, 256], [vline, vline], 'Color', 'r', 'Linewidth', 1);
% end
% hold off;

%% extract lattices from separator
[M, N] = size(cartoon);
if (Sv(end) + Lv) <= M && (Sh(end) + Lh) <= N
    lattices = zeros(length(Sv), length(Sh), Lv + 1, Lh + 1);
else
    lattices = zeros(length(Sv) - 1, length(Sh) - 1, Lv + 1, Lh + 1);
end
for i = 1:length(Sv)
    for j = 1:length(Sh)
        y_start = Sv(i); x_start = Sh(j);
        y_end = Sv(i) + Lv; x_end = Sh(j) + Lh;
        if y_end > M || x_end > N, continue, end
%         lattices(i, j, :, :) = img(y_start:y_end, x_start:x_end, :);
        lattices(i, j, :, :) = cartoon(y_start:y_end, x_start:x_end, :);
    end
end

%% generate distance matrix
% s = 1;
% wavelength = 2^(2 + s / 2);
% oris = [0 45 90 135];
oris = [-180:90:180];
% g = gabor(wavelength, orientations, 'SpatialFrequencyBandwidth', 0.7, 'SpatialAspectRatio', 1);
% imshow(imag(g(2).SpatialKernel),[]);colormap(parula);
V = zeros(size(lattices, 1), size(lattices, 2), length(oris), Lh + 1 + Lv + 1);
for i = 1:size(lattices, 1)
    for j = 1:size(lattices, 2)
        latt = squeeze(lattices(i, j, :, :));
        for n_ori = 1:length(oris)
            gbr = gabor_filter(4, 1, oris(n_ori));
            G_ij = imfilter(latt, gbr, 'symmetric');
            G_ij = imrotate(G_ij, -oris(n_ori), 'bilinear', 'crop');
            a = sum(G_ij, 1); b = sum(G_ij, 2);
            V(i, j, n_ori, :) = [a(:); b(:)];
        end
    end
end
D = dist_matrix(V);
% figure; imagesc(D); colormap(jet);

%% return distance matrix or binary mask result
if nargin == 1
    return;
else
    % d_star = 0.3;
    [counts, centers] = hist(D(:));
    tt = find(centers > d_star, 1);
    if tt > 1
        counts = counts(tt - 1:end);
        centers = centers(tt - 1:end);
    end
    
    t_star = d_star;
    % find value for t''
    for tpp_ind = 2:length(counts)
        if counts(tpp_ind) > 2 * counts(tpp_ind - 1)
            t_star = centers(tpp_ind);
            break;
        end
    end
    
    % find value for t'
    tp_ind = find(counts > 0, 1);
    if ~isempty(tp_ind) && tp_ind > 1 && counts(tp_ind - 1) > 0
        t_star = centers(tp_ind);
    end
    
    [row_inds, col_inds] = ind2sub(size(D), find(D > t_star));
    
    binary = zeros(size(img), 'like', img);
    for i = 1:length(row_inds)
        for j = 1:length(col_inds)
            y_start = Sv(row_inds(i)); x_start = Sh(col_inds(j));
            y_end = Sv(row_inds(i)) + Lv; x_end = Sh(col_inds(j)) + Lh;
            if y_end > M || x_end > N, continue, end
            binary(y_start:y_end, x_start:x_end, :) = 1;
        end
    end
    varargout{1} = binary;
    return;
end

end
