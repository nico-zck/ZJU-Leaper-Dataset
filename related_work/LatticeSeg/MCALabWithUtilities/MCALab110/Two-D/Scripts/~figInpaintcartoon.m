%%%%%%%%%%%%%%% Barbara image %%%%%%%%%%%%%%%%%%%%%%%%
clear all
img  = double(imread('tomandjerry_fingerprint.png'));
mask = double(imread('Mask.png'));mask=1-normsignal(mask,0,1);
img  = img(1:2:end,1:2:end);
mask = mask(1:2:end,1:2:end);

% Dictionary stuff (here Curvelets + UDWT).
qmf=MakeONFilter('Symmlet',6);
dict1='CURVWRAP';pars11='2';pars12=0;pars13=0;
dict2='WAVEATOM2';pars21='q';pars22=0;pars23=0;
%dict2='LDCT2iv';pars21='Sine';pars22=length(img)/2;pars23=0.25;
dicts=MakeList(dict1,dict2);
pars1=MakeList(pars11,pars21);
pars2=MakeList(pars12,pars22);
pars3=MakeList(pars13,pars23);


% MCA inpainting.
itermax		= 300;
tvregparam 	= 10;
tvcomponent	= 1;
expdecrease	= 1;
lambdastop	= 1E-4;
display		= 1;

imasked = img.*mask;
[partsMCA,optionsMCA]=MCA2_Bcr(imasked,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,[],display);
optionsMCA.inputdata = 'Input image: Cartoon';
optionsMCA.maskdata  = sprintf('Mask image');
optionsMCA
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s optionsMCA -V6',[name(1:end-2) 'metadata']));

% Display results.
figure;
set(gcf,'Name','Inpainting Cartoon','NumberTitle','off');
subplot(211);
imagesc(img);axis image;rmaxis;
title('Original Barbara');

subplot(223);
imagesc(imasked);axis image;rmaxis;
title(sprintf('Masked'));
 
subplot(224);
imagesc(squeeze(sum(partsMCA,3)));axis image;rmaxis;
title(sprintf('Inpainted'));

colormap('gray');

[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')'])
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));
