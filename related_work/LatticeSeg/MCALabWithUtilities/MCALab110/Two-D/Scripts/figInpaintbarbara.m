%%%%%%%%%%%%%%% Barbara image %%%%%%%%%%%%%%%%%%%%%%%%
clear all
img = double(imread('barbara_512x512.png'));
masks = {'mask_512x512_random20.bmp','mask_512x512_random50.bmp','mask_512x512_random80.bmp'};
nbmis = [20 50 80];

% Dictionary stuff (here Curvelets + UDWT).
qmf=MakeONFilter('Symmlet',6);
dict1='CURVWRAP';pars11='2';pars12=0;pars13=0;
dict2='LDCT2iv';pars21='Sine';pars22=32;pars23=0.25;
dicts=MakeList(dict1,dict2);
pars1=MakeList(pars11,pars21);
pars2=MakeList(pars12,pars22);
pars3=MakeList(pars13,pars23);


% MCA inpainting.
itermax		= 300;
epsilon		= 1E-3;
lambda		= 0;
sigma		= 1;
thdtype		= 'Hard';
ecmtype		= 2;
tvregparam 	= 0;
tvcomponent	= 0;
expdecrease	= 0;
lambdastop	= 0;
display		= 1;
for i=1:length(masks)
 disp(sprintf('Mask with %d%% missing pixels',nbmis(i))); 
 mask = double(imread(masks{i}));mask=mask(:,:,1);mask(mask~=0)=1;
 imasked = img.*mask;
 %[partsECMDA(:,:,:,i),optionsECM(i)]=EM_Inpaint(imasked,dicts,pars1,pars2,pars3,itermax,epsilon,lambda,sigma,thdtype,ecmtype,mask);
 [partsMCA(:,:,:,i),optionsMCA{i}]=MCA2_Bcr(imasked,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,[],display);
 optionsMCA{i}.inputdata = 'Input image: Barbara 512 x 512';
 optionsMCA{i}.maskdata  = sprintf('Mask image: random with %.2f%% missing pixels',nbmis(i));
 options=optionsMCA{i}
 [ST,I] = dbstack;
 name=eval(['which(''' ST(1).name ''')']);
 eval(sprintf('save %s_%dmetadata options -V6',name(1:end-2),nbmis(i)));
end

% Display results.
figure;
set(gcf,'Name','Inpainting Barbara','NumberTitle','off');
subplot(411);
imagesc(img);axis image;rmaxis;
title('Original Barbara');

for i=1:length(masks)
 mask = double(imread(masks{i}));mask=mask(:,:,1);mask(mask~=0)=1;
 imasked = img.*mask;
 iminptd = squeeze(sum(partsMCA(:,:,:,i),3));
 
 subplot(4,2,2*i+1);
 imagesc(imasked);axis image;rmaxis;
 title(sprintf('Masked %d%% missing pixels',nbmis(i)));
 
 subplot(4,2,2*i+2);
 imagesc(iminptd);axis image;rmaxis;
 title(sprintf('Inpainted Barbara PSNR=%.3g dB',psnr(img,iminptd)));
end

colormap('gray');


