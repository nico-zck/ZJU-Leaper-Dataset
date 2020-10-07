%%%%%%%%%%%%%%% Barbara image %%%%%%%%%%%%%%%%%%%%%%%%
clear all
imgt = 0.4*double(imread('texture4.tif'));
imgc = 0.6*double(imread('boy.tif'));

% Observed image.
img = imgt + imgc;

% Dictionary stuff (here Curvelets + UDWT).
dict1='CURVWRAP';pars11=2;pars12=0;pars13=0;
dict2='LDCT2';pars21=256;pars22=16/256;pars23=0; % Remove Low-frequencies < 16/256 from textured part.
dicts=MakeList(dict1,dict2);
pars1=MakeList(pars11,pars21);
pars2=MakeList(pars12,pars22);
pars3=MakeList(pars13,pars23);


% Call the MCA.
itermax 	= 50;
tvregparam 	= 0.1;
tvcomponent	= 1;
expdecrease	= 1;
lambdastop	= 1;
sigma		= 1E-6;
display		= 1;
[parts,options]=MCA2_Bcr(img,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,[],sigma,display);
options.inputdata = 'Input image: Boy + Texture 256 x 256';
options
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));

% Display results.
figure;
set(gcf,'Name','MCA Texture + Boy','NumberTitle','off');
subplot(321);
imagesc(img);axis image;rmaxis;
title('Original Texture + Boy');

subplot(322);
imagesc(squeeze(sum(parts,3)));axis image;rmaxis;
title(sprintf('Recovered MCA PSNR=%.3g dB',psnr(img,squeeze(sum(parts,3)))));

subplot(323);
imagesc(imgc);axis image;rmaxis;
title('Original Cartoon part');

subplot(324);
imagesc(squeeze(parts(:,:,1)));axis image;rmaxis;
title(sprintf('MCA Cartoon'));

subplot(325);
imagesc(imgt);axis image;rmaxis;
title('Original Texture');

subplot(326);
imagesc(squeeze(parts(:,:,2)));axis image;rmaxis;
title(sprintf('MCA Texture'));

colormap('gray');


