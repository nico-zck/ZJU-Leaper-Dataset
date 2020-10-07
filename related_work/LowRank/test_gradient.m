im = im2double(rgb2gray(imread("bad_tk4.png")));

methods = ["sobel", "prewitt", "central", "intermediate", "roberts"];
n_method = length(methods);

%% original
img = im;
figure;
subplot(1, n_method + 1, 1);
imshow(img);

for i = 1:n_method
    subplot(1, n_method + 1, i + 1);
    [Gmag, Gdir] = imgradient(img, methods(i));
    imshow(exp(-Gmag), []);
end

%% original
img = histeq(im);
figure;
subplot(1, n_method + 1, 1);
imshow(img);

for i = 1:n_method
    subplot(1, n_method + 1, i + 1);
    [Gmag, Gdir] = imgradient(img, methods(i));
    imshow(exp(-Gmag), []);
end

%% original
img = imsharpen(im);
figure;
subplot(1, n_method + 1, 1);
imshow(img);

for i = 1:n_method
    subplot(1, n_method + 1, i + 1);
    [Gmag, Gdir] = imgradient(img, methods(i));
    imshow(exp(-Gmag), []);
end

%% original
img = imadjust(im);
figure;
subplot(1, n_method + 1, 1);
imshow(img);

for i = 1:n_method
    subplot(1, n_method + 1, i + 1);
    [Gmag, Gdir] = imgradient(img, methods(i));
    imshow(exp(-Gmag), []);
end

%% original
img = brighten(im);
figure;
subplot(1, n_method + 1, 1);
imshow(img);

for i = 1:n_method
    subplot(1, n_method + 1, i + 1);
    [Gmag, Gdir] = imgradient(img, methods(i));
    imshow(exp(-Gmag), []);
end
