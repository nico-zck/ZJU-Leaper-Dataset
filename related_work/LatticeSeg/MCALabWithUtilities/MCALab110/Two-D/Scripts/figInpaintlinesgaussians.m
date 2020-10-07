%%%%%%%%%%%%%%% 2D synthetic image: lines+gaussians %%%%%%%%%%%%%%%%%%%%%%%%
clear all
load linesgaussians
mask=double(imread('mask_texturegaussians.bmp'));mask=mask(:,:,1);mask(mask~=0)=1;

% Image to inpaint.
imasked=img.*mask;

% Dictionary stuff (here Curvelets + UDWT).
qmf=MakeONFilter('Symmlet',6);
dict1='UDWT2';pars11=3;pars12=qmf;pars13=0;
dict2='CURVWRAP';pars21=2;pars22=0;pars23=0;
dicts=MakeList(dict1,dict2);
pars1=MakeList(pars11,pars21);
pars2=MakeList(pars12,pars22);
pars3=MakeList(pars13,pars23);


% EM inpaiting: ECM with fixed lambda.
epsilon		= 1E-6;
lambda		= 0.3;
sigma		= 1;
thdtype		= 'Soft';
ecmtype		= 0;
display		= 1;
[imginpECM,optionsECM]=EM_Inpaint(imasked,dicts,pars1,pars2,pars3,1,epsilon,lambda,sigma,thdtype,ecmtype,mask,display);
optionsECM.inputdata = 'Input image: Lines + Gaussians 256 x 256';
optionsECM.maskdata  = sprintf('Mask image: Brush strokes mask with %.2f%% missing pixels',100*length(find(mask(:)))/prod(size(img)));
optionsECM
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s optionsECM -V6',[name(1:end-2) 'ECMmetadata']));

% MCA inpainting: varying lambda.
itermax 	= 50;
tvregparam 	= 0;
tvcomponent	= 0;
expdecrease	= 0;
lambdastop	= 0;
display		= 1;
[partsMCA,optionsMCA]=MCA2_Bcr(imasked,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,[],display);
optionsMCA.inputdata = 'Input image: Lines + Gaussians 256 x 256';
optionsMCA.maskdata  = sprintf('Mask image: Brush strokes mask with %.2f%% missing pixels',100*length(find(mask(:)))/prod(size(img)));
optionsMCA
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s optionsMCA -V6',[name(1:end-2) 'MCAmetadata']));

% Display results.
figure;
set(gcf,'Name','Inpainting Lines + Gaussians','NumberTitle','off');
subplot(221);
imagesc(img);axis image;rmaxis;
title('Original Lines + Gaussians');

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


