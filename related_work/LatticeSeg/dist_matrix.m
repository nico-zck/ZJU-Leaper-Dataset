function [D] = dist_matrix(V)

% dist_func = 'chebychev';
% dist_func = 'euclidean';
dist_func = 'cosine';
% dist_func = 'spearman';

[m, n, d, l] = size(V);

row_best_ind = [];
for mi = 1:m
    D_row = zeros(d, n, n);
    V_i = squeeze(V(mi, :, :, :));
    inds = [];
    for di = 1:d
        V_i_d = squeeze(V_i(:, di, :));
        D_i_d = pdist2(V_i_d, V_i_d, dist_func);
        D_row(di, :, :) = D_i_d;
        [~, j] = min(std(D_i_d));
        inds(di) = j;
    end
%     if mi == 7
%         D_row = squeeze(mean(D_row, 1));
%         figure; imagesc(D_row); colormap(jet);
%     end
    row_best_ind(mi) = mode(inds);
end

V_bests = zeros(m, d, l);
for i = 1:length(row_best_ind)
    V_bests(i, :, :) = V(i, row_best_ind(i), :, :);
end

D_p = zeros(d, m, m);
for di = 1:d
    V_best_d = squeeze(V_bests(:, di, :));
    D_p(di, :, :) = pdist2(V_best_d, V_best_d, dist_func);
end
D_p = squeeze(mean(D_p, 1));

d_vec = sum(D_p, 2);
d_mean = mean(d_vec);
d_std = std(d_vec);

V_bests = V_bests(d_vec < (d_mean + d_std) & d_vec > (d_mean - d_std), :, :);
V_best = squeeze(mean(V_bests, 1));

D = zeros(d, m * n);
V_flat = reshape(V, m * n, d, l);
for di = 1:d
    D(di, :) = pdist2(squeeze(V_flat(:, di, :)), squeeze(V_best(di, :)), dist_func);
end
D = squeeze(mean(D, 1));
D = reshape(D, m, n);

end
