%%%%%%%%%%%%%%% Lena image %%%%%%%%%%%%%%%%%%%%%%%%
clear all
img = double(imread('lena512x512.png'));
mask=double(imread('mask_text_512x512.bmp'));mask=mask(:,:,1);mask(mask~=0)=1;mask=1-mask;

%img = img(1:2:end,1:2:end);
%mask= mask(1:2:end,1:2:end);

% Image to inpaint.
imasked=img.*mask;

% Dictionary stuff (here Curvelets + UDWT).
dict1='CURVWRAP';pars11=2;pars12=0;pars13=0;

% EM inpaiting: ECM with fixed lambda.
epsilon = 1E-6;
lambda = 10;
sigma = 1;
thdtype = 'Soft';
ecmtype = 0;
display = 1;
[imginpECM,optionsECM]=EM_Inpaint(imasked,dict1,pars11,pars12,pars13,1,epsilon,lambda,sigma,thdtype,ecmtype,mask,display);
optionsECM.inputdata = 'Input image: Lena 512 x 512';
optionsECM.maskdata  = sprintf('Mask image: Text mask with %.2f%% missing pixels',100*length(find(mask(:)))/prod(size(img)));
optionsECM
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s optionsECM -V6',[name(1:end-2) 'ECMmetadata']));

% MCA inpainting: varying lambda.
itermax 	= 300;
tvregparam 	= 0;
tvcomponent	= 0;
expdecrease	= 1;
lambdastop	= 1E-6;
display		= 1;
[partsMCA,optionsMCA]=MCA2_Bcr(imasked,dict1,pars11,pars12,pars13,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,1,display);
optionsMCA.inputdata = 'Input image: Lena 512 x 512';
optionsMCA.maskdata  = sprintf('Mask image: Text mask with %.2f%% missing pixels',100*length(find(mask(:)))/prod(size(img)));
optionsMCA
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s optionsMCA -V6',[name(1:end-2) 'MCAmetadata']));

% Display results.
figure;
set(gcf,'Name','Inpainting Lena','NumberTitle','off');
subplot(221);
imagesc(img);axis image;rmaxis;
title('Original Lena');

subplot(222);
imagesc(imasked);axis image;rmaxis;
title(sprintf('Masked PSNR=%.3g dB',psnr(img,imasked)));

subplot(223);
imagesc(imginpECM);axis image;rmaxis;
title(sprintf('Inpainted ECM PSNR=%.3g dB',psnr(img,imginpECM)));

subplot(224);
imagesc(squeeze(sum(partsMCA,3)));axis image;rmaxis;
title(sprintf('Inpainted MCA PSNR=%.3g dB',psnr(img,squeeze(sum(partsMCA,3)))));

colormap('gray');


