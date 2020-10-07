% image
D = im2double(rgb2gray(imread('bad_tk4.png')));
% D = im2double(rgb2gray(imread('Bbox_bn_d5.bmp')));

[A1, E1] = LRD(D, 0.1);

figure;
subplot(141); imshow(A1, []);
subplot(142); imshow(abs(E1), []);
subplot(143); imshow(E1, []);
