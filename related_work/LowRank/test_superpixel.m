im=im2double(rgb2gray(imread('bad_tk4.png')));

mr8=MR8fast(im);
m=sqrt(size(mr8, 2));
mr8=reshape(mr8, 8, m, m);
figure;
for i = 1 : 8
   subplot(1,8,i);
   imshow(squeeze(mr8(i,:,:)),[]);
end

A=im;
[L,N] = superpixels(A,500);
spim = zeros(size(A),'like',A);
idx = label2idx(L);
for labelVal = 1:N
    redIdx = idx{labelVal};
    spim(redIdx) = mean(A(redIdx));
end    

figure;imshow(spim,[]);

spmr8=MR8fast(spim);
m=sqrt(size(spmr8, 2));
spmr8=reshape(spmr8, 8, m, m);
figure;
for i = 1 : 8
   subplot(1,8,i);
   imshow(squeeze(spmr8(i,:,:)),[]);
end
