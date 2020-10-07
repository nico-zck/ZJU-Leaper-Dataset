%%%%%%%%%%%%%%% 2D synthetic image: texture+gaussians %%%%%%%%%%%%%%%%%%%%%%%%
clear all
n=256;
% Make three patches of textured part.
ix = ((-n/2):(n/2-1))' * ones(1,n);
iy = ones(n,1) * ((-n/2):(n/2-1));
ix = ix./n; iy = iy./n;
imgt=zeros(n,n);
imgt(1:n/4,1:n/4)=cos(2*pi*ix(1:n/4,1:n/4)*30).*cos(2*pi*iy(1:n/4,1:n/4)*30);
imgt(1:n/4,n-n/4+1:n)=cos(2*pi*ix(1:n/4,1:n/4)*30+2*pi*iy(1:n/4,1:n/4)*30);
imgt(n/2-n/8+1:n/2+n/8,n/2-n/8+1:n/2+n/8)=cos(2*pi*ix(1:n/4,1:n/4)*30).*sin(2*pi*iy(1:n/4,1:n/4)*15);

% Make the gaussian part.
imgg=exp(-160.*ix.^2- 160.*iy.^2 )+exp(-160.*(ix-0.25).^2- 160.*(iy-0.25).^2)+exp(-640.*(ix-0.25).^2-640.*(iy+0.25).^2);

% Reduce to half size for fast demo (comment if undesired).
imgt=imgt(1:2:end,1:2:end);
imgg=imgg(1:2:end,1:2:end);

% Original image.
img =imgt+imgg;

% imnoisy = part1 + part2 + noise
sigma  = 0.1;
imnoisy=img+sigma*randn(size(img));

% Dictionary stuff (here local DCT + UDWT).
qmf=MakeONFilter('Symmlet',6);
dict1='UDWT2';pars11=2;pars12=qmf;pars13=0;
dict2='LDCT2';pars21=16;pars22=0;pars23=0;
dicts=MakeList(dict1,dict2);
pars1=MakeList(pars11,pars21);
pars2=MakeList(pars12,pars22);
pars3=MakeList(pars13,pars23);


% Call the MCA.
itermax 	= 50;
tvregparam 	= 2;
tvcomponent	= 1;
expdecrease	= 0;
lambdastop	= 3;
display		= 1;
[parts,options]=MCA2_Bcr(imnoisy,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,[],[],display);
options.inputdata = 'Input image: Texture + Gaussians';
options
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));

% Display results.
figure;
set(gcf,'Name','MCA Texture + Gaussians','NumberTitle','off');
subplot(331);
imagesc(img);axis image;rmaxis;
title('Original Texture + Gaussians');

subplot(332);
imagesc(imnoisy);axis image;rmaxis;
title(sprintf('Noisy PSNR=%.3g dB',psnr(img,imnoisy)));

subplot(333);
imagesc(squeeze(sum(parts,3)));axis image;rmaxis;
title(sprintf('Denoised MCA PSNR=%.3g dB',psnr(img,squeeze(sum(parts,3)))));

subplot(323);
imagesc(imgg);axis image;rmaxis;
title('Original Gaussians');

subplot(324);
imagesc(squeeze(parts(:,:,1)));axis image;rmaxis;
title(sprintf('MCA Gaussians PSNR=%.3g dB',psnr(imgg,squeeze(parts(:,:,1)))));

subplot(325);
imagesc(imgt);axis image;rmaxis;
title('Original Texture');

subplot(326);
imagesc(squeeze(parts(:,:,2)));axis image;rmaxis;
title(sprintf('MCA Texture PSNR=%.3g dB',psnr(imgt,squeeze(parts(:,:,2)))));

colormap('gray');


