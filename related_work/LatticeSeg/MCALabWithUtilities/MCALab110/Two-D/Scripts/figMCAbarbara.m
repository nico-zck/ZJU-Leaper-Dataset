%%%%%%%%%%%%%%% Barbara image %%%%%%%%%%%%%%%%%%%%%%%%
clear all
img = double(imread('barbara_512x512.png'));

% Dictionary stuff (here Curvelets + UDWT).
dict1='CURVWRAP';pars11='2';pars12=0;pars13=0;
dict2='LDCT2iv';pars21='Sine';pars22=32;pars23=128/512; % Remove Low-frequencies 128/512 from textured part.
dicts=MakeList(dict1,dict2);
pars1=MakeList(pars11,pars21);
pars2=MakeList(pars12,pars22);
pars3=MakeList(pars13,pars23);


% Call the MCA.
itermax 	= 300;
tvregparam 	= 2;
tvcomponent	= 1;
expdecrease	= 1;
lambdastop	= 1;
sigma		= 1E-6;
display		= 1;
[parts,options]=MCA2_Bcr(img,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,[],sigma,display);
options.inputdata = 'Input image: Barbara 512 x 512';
options
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));

% Display results.
figure;
set(gcf,'Name','MCA Barbara','NumberTitle','off');
subplot(221);
imagesc(img);axis image;rmaxis;
title('Original Barbara');

subplot(222);
imagesc(squeeze(sum(parts,3)));axis image;rmaxis;
title(sprintf('MCA Barbara PSNR=%.3g dB',psnr(img,squeeze(sum(parts,3)))));

subplot(223);
imagesc(squeeze(parts(:,:,1)));axis image;rmaxis;
title('Barbara Cartoon');

subplot(224);
imagesc(squeeze(parts(:,:,2)));axis image;rmaxis;
title('Barbara Texture');

colormap('gray');


