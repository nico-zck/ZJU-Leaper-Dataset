% Signal.
n=512;	% Signal length.
x = GenSignal('Star');
x = x(1:n);

% Observed sgnal = part1 + part2
y = x;

% Representation dictionary: overcomplete LDCTiv+DIRAC.
dicts = MakeList('LDCTiv','DIRAC');
pars1 = MakeList('Sine',0);
pars2 = MakeList(n/8,0);
pars3 = MakeList(0,0);

% No inpainting => mask of ones or empty mask.
mask = [];


% Call the MCA.
itermax 	= 100;
tvregparam 	= 0;
tvcomponent	= 0;
expdecrease	= 0;
lambdastop	= 0;
display		= 1;
[parts,options]=MCA_Bcr(y,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,[],display);
options
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
set(gcf,'Name','MCA 1D Star signal','NumberTitle','off');
subplot(4,1,1);
plot(sum(parts,2));hold on;plot(x,'--');hold off;axis tight;
legend('Recovered','Original');
xlabel('Time');
title('(a)');

subplot(4,1,2);
plot(parts(:,1));axis tight;
xlabel('Time');
title(sprintf('(b) \n MCA LDCT component'));

subplot(4,1,3);
C = FastLA(parts(:,1),NthList(dicts,1),NthList(pars1,1),NthList(pars2,1),NthList(pars3,1));C = C{1};
plot(C(2).coeff);
axis tight;
xlabel('Frequency');
title(sprintf('(c) \n MCA LDCT coeffs'));

subplot(4,1,4); 
plot(1:n,parts(:,end));axis tight;
xlabel('Time');
title(sprintf('(d) \n MCA Dirac component'));

