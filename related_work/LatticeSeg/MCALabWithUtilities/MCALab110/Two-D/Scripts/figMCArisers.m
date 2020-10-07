%%%%%%%%%%%%%%% 2D synthetic image: lines+texture %%%%%%%%%%%%%%%%%%%%%%%%
clear all
imnoisy=double(imread('risers.bmp'));

% Dictionary stuff (here Curvelets + UDWT).
qmf=MakeONFilter('Symmlet',6);
dict1='UDWT2';pars11=2;pars12=qmf;pars13=0;
dict2='CURVWRAP';pars21=2;pars22=0;pars23=0;
dicts=MakeList(dict1,dict2);
pars1=MakeList(pars11,pars21);
pars2=MakeList(pars12,pars22);
pars3=MakeList(pars13,pars23);


% Call the MCA.
itermax 	= 30;
tvregparam 	= 3;
tvcomponent	= 1;
expdecrease	= 1;
lambdastop	= 3;
display		= 1;
[parts,options]=MCA2_Bcr(imnoisy,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,[],[],display);
options.inputdata = 'Input image: Risers 150 x 501';
options
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));

% Display results.
figure;
set(gcf,'Name','MCA Risers','NumberTitle','off');
subplot(311);
imagesc(imnoisy);axis image;rmaxis;
title('Original');

subplot(312);
imagesc(squeeze(parts(:,:,1)));axis image;rmaxis;
title('MCA Isotropic');

subplot(313);
imagesc(squeeze(parts(:,:,2)));axis image;rmaxis;
title('MCA Lines');


colormap('gray');


