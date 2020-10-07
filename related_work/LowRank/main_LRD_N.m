% image
D = im2double(rgb2gray(imread('bad_tk4.png')));
% D = im2double(rgb2gray(imread('bstar_tk_d3.bmp')));

[A2, E2, N2] = LRD_N(D, 0.03, 0.2);

figure;
subplot(141); imshow(A2, []);
subplot(142); imshow(abs(E2), []);
subplot(143); imshow(abs(N2), []);
subplot(144); imshow(E2, []);
