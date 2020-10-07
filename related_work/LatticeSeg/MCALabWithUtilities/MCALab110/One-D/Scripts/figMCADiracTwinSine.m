n=1024;	% Signal length.

% Signal.
% Four cosines have close frequencies, not in the dictionary
fineness = 4;
t = ((1:n)' - .5) / n;freq = pi * ((1:(4*n))' - 1) / (fineness*n);
freq1 = pi * (126.55-1) / (fineness*n);
freq2 = pi * (127.55-1) / (fineness*n);
freq3 = pi * (128.55-1) / (fineness*n);
freq4 = pi * (129.55-1) / (fineness*n);
const = (2/n) ^ .5;
x1 = const * cos(pi * ((126.55  - 1) / fineness) * t);
x2 = const * cos(pi * ((127.55  - 1) / fineness) * t);
x3 = const * cos(pi * ((128.55  - 1) / fineness) * t);
x4 = const * cos(pi * ((129.55  - 1) / fineness) * t);
zerosn = zeros(n,1);
EnergyRatio = 1E1;
xdct = EnergyRatio*(x1 + x2 + x3 + x4);
xdir = SparseVector(n, 3);
xdir(find(xdir)) = normsignal(xdir(find(xdir)),1,2);
pos = find(xdir);
x = xdct + xdir;

% Observed data.
y = x;

% Representation dictionary: overcomplete DCT+DIRAC.
dicts = MakeList('DCT','DIRAC');
pars1 = MakeList(fineness,0);
pars2 = MakeList(0,0);
pars3 = MakeList(0,0);

% No inpainting => mask of ones or empty mask.
mask = [];


% Call the MCA.
itermax 	= 200;
tvregparam 	= 0;
tvcomponent	= 0;
expdecrease	= 0;
lambdastop	= 0;
display		= 1;
[parts,options]=MCA_Bcr(y,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,[],display);
options.inputdata = 'Input signal: Dirac+TwinSine';
options
[ST,I] = dbstack;
name=eval(['which(''' ST(1).name ''')']);
eval(sprintf('save %s options -V6',[name(1:end-2) 'metadata']));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
set(gcf,'Name','MCA 1D Dirac + TwinSine','NumberTitle','off');
subplot(4,1,1);
plot(sum(parts,2));hold on;plot(x,'--','LineWidth',2);hold off;axis tight;
xlabel('Time');
legend('Recovered','Original','Location','Best');
title(sprintf('(a) \n MCA Recovered PSNR=%g dB',psnr(x(:),sum(parts,2))));

subplot(4,1,2);
plot(parts(:,1));hold on;plot(xdct,'--','LineWidth',2);hold off;axis([1 n min(xdct) max(xdct)]);
xlabel('Time');
legend('Recovered','Original','Location','Best');
title(sprintf('(b) \n MCA TwinSine component PSNR=%g dB',psnr(xdct(:),parts(:,1))));

subplot(4,1,3);
C = FastLA(parts(:,1),NthList(dicts,1),NthList(pars1,1),NthList(pars2,1),NthList(pars3,1));C = C{1}; 
stem(freq,abs(C(2).coeff),'.');
xlabel('Frequency');
axis1 = axis;
axis([freq(120) freq(140) axis1(3) axis1(4)]);
X1 = [freq1 freq2 freq3 freq4]; X1 = [X1;X1];
Y1 = [axis1(3)*ones(1,4);axis1(4)*ones(1,4)];
hold on
plot(X1, Y1, ':b');
hold off
title(sprintf('(c) \n MCA DCT coeffs'));

subplot(4,1,4); 
plot(1:n,parts(:,2),'-b',pos,xdir(pos),'+r');axis tight;
xlabel('Time');
title(sprintf('(d) \n MCA Dirac component PSNR=%g dB',psnr(xdir(:),parts(:,2))));


