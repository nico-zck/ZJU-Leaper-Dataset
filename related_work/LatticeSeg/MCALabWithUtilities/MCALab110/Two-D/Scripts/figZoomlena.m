%%%%%%%%%%%%%%% Lena image 256 x 256 %%%%%%%%%%%%%%%%%%%%%%%%
clear all
factor = 2; % Subsampling factor.
img = ReadImage('Lenna');
mask=zeros(size(img));mask(1:factor:end,1:factor:end)=1;

% Image to zoom.
imasked=img.*mask;

% Dictionary stuff (here Curvelets + UDWT).
dict1='CURVWRAP';pars11=2;pars12=0;pars13=0;

% EM inpaiting: ECM with fixed lambda.
epsilon = 1E-6;
lambda = 4;
sigma = 1;
thdtype = 'Soft';
ecmtype = 0;
display = 1;
[imginpECM,optionsECM]=EM_Inpaint(imasked,dict1,pars11,pars12,pars13,1,epsilon,lambda,sigma,thdtype,ecmtype,mask,display);
optionsECM.inputdata = 'Input image: Lena 256 x 256';
optionsECM.maskdata  = sprintf('Mask image: Downsampling by factor 2 x 2');

% MCA inpainting: varying lambda.
itermax 	= 100;
tvregparam 	= 0;
tvcomponent	= 0;
expdecrease	= 1;
lambdastop	= 1E-6;
display		= 1;
[partsMCA,optionsMCA]=MCA2_Bcr(imasked,dict1,pars11,pars12,pars13,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,1,display);
optionsMCA.inputdata = 'Input image: Lena 256 x 256';
optionsMCA.maskdata  = sprintf('Mask image: Downsampling by factor 2 x 2');

% Display results.
figure;
set(gcf,'Name','Zooming Lena','NumberTitle','off');
subplot(221);
imagesc(img);axis image;rmaxis;
title('Original Lena');

subplot(222);
imagesc(img(1:factor:end,1:factor:end));axis image;rmaxis;
title(sprintf('Downsampled by a factor %d x %d',factor,factor));

subplot(223);
imagesc(imginpECM);axis image;rmaxis;
title(sprintf('Zoomed ECM PSNR=%.3g dB',psnr(img,imginpECM)));

subplot(224);
imagesc(squeeze(sum(partsMCA,3)));axis image;rmaxis;
title(sprintf('Zoomed MCA PSNR=%.3g dB',psnr(img,squeeze(sum(partsMCA,3)))));

colormap('gray');


