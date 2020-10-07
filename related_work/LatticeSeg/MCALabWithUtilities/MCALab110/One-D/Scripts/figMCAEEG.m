% Signal.
x = GenSignal('EegfMRI');
n = length(x);	% Signal length.

% Observed sgnal = part1 + part2
y = x;

% Representation dictionary (here local DCT + UDWT + Dirac).
qmf=MakeONFilter('Symmlet',6);
dicts = MakeList('LDCT','UDWT');
pars1 = MakeList(32,2);
pars2 = MakeList(0.5,qmf);
pars3 = MakeList(0,5);

% No inpainting => mask of ones or empty mask.
mask = [];


% Call the MCA.
itermax 	= 100;
tvregparam 	= 0;
tvcomponent	= 0;
expdecrease	= 1;
lambdastop	= 4;
display		= 1;
[parts,options]=MCA_Bcr(y,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,[],display);
options.inputdata = 'Input signal: EEG + fMRI medical signal';
options
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
set(gcf,'Name','MCA 1D EEG-fMRI signal','NumberTitle','off');
subplot(4,1,1);
plot(y);axis tight
xlabel('Time');
title(sprintf('(a) \n EEG-fMRI signal'));

subplot(4,1,2);
plot(sum(parts,2));axis tight
xlabel('Time');
title(sprintf('(b) \n Denoised'));

subplot(4,1,3); 
plot(parts(:,1));axis tight;
xlabel('Time');
title(sprintf('(c) \n MCA MRI magnetic field induced component'));

subplot(4,1,4); 
plot(parts(:,2));axis tight;
xlabel('Time');
title(sprintf('(d) \n MCA EEG MRI-free component'));


