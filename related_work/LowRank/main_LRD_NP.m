% image
D = im2double(rgb2gray(imread('bad_tk4.png')));
% D = im2double(rgb2gray(imread('bstar_tk_d3.bmp')));

% defect prior
P = def_prior(D, false);
P = (P-min(P(:))) / (max(P(:))-min(P(:)));

P = im2double(imread('bad_tk4_mask.png'));
% P = im2double(imread('groundT_Bbox_bn_d5.bmp'));

[A3, E3, N3] = LRD_NP(D, P, 0.03, 0.2);

figure;
subplot(141); imshow(A3, []);
subplot(142); imshow(abs(E3), []);
subplot(143); imshow(abs(N3), []);
subplot(144); imshow(E3, []);

S = abs(E3);
S = S .* S;
th = adaThresh(S);
BS = S > th;
figure;imshow(BS, []);