% image
D = im2double(rgb2gray(imread('bad_tk4.png')));
% D = im2double(rgb2gray(imread('bstar_tk_d3.bmp')));

% defect prior
P = def_prior(D, false);
P = (P-min(P(:))) / (max(P(:))-min(P(:)));
% P = im2double(imread('bad_tk4_mask.png'));

% gradient information
G = grad_info(D);

[A4, E4, N4] = LRD_NPG(D, P, G, 0.03, 0.2);
E4_ = abs(E4);

figure;
subplot(141); imshow(A4, []);
subplot(142); imshow(E4, []);
subplot(143); imshow(E4_, []);
subplot(144); imshow(N4, []);
