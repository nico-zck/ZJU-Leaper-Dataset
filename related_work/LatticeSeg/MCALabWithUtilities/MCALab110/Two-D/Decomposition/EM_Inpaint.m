function [imginp,options]=EM_Inpaint(img,dict,pars1,pars2,pars3,itermax,epsilon,lambda,sigma,thdtype,ecmtype,mask,display)
% EM_Inpaint: Bayesian Inpainting of 2D images (a matrix) using redundant dictionaries.
%	   The optimization pb is solved using the EM algorithm.
%	   EM_Inpaint solves the following (MAP) optimization problem
%		(part) = argmin lambda * Psi(coeff) +  ||img - M \Phi coeff||_2^2/(2 sigma^2)
%				M is the mask.
%				Phi is the dictionary.
%				Psi is any penalty on the dictionary coefficients (hopefully a convex function).
%	   The image is supposed to be sparsely described in the dictionary \Phi.
%  Usage:
%    part=EM_Inpaint(img,dict,pars1,pars2,pars3,itermax,mask)
%  Inputs:
%    img	     	2D image matrix, nxn, n = 2^J
%    dict		Names of dictionaries for each part (see directory Dictionary for available transforms)
%    parsi		Parameters of dictionary (built using the MakeList function)
%    itermax		Nb of iterations
%    epsilon		Stops when residual l_2-error change is < epsilon
%    lambda		Regularization parameter (not always required, see the EM algo type argument)
%    sigma		Standard deviation of the noise (can be estimated and updated automatically)
%    thdtype		Penalty (threshold) type
%    ecmtype		Type of ECM algorithm (0: ECM, 1: ECM with std estimation annealing, 2: Deterministic Annealing ECM). 
%			If emtype == 2 then argument lambda is discarded.
%    mask		The binary mask.
%    display		Display algorithm progress. 
%  Outputs:
%    imginp 		Estimated inpainted image (n x n image)
%    options		Structure containing all options and parameters of the algorithm, including default choices.
%
%  Description
%    The dictionaries and their parameters can be built using the MakeList function.
%  See Also
%    FastLA2, FastLS2, MCA2_Bcr

global E % Energy (L2 norm) of curvelets if CURVWRAP is used in the dictionary

% Initializations.
	[N,M]	= size(img);
	n	= 2^(nextpow2(max(N,M)));
	imgpad	= zeros(n,n);
	maskpad	= zeros(n,n);
	imgpad(1:N,1:M)  = img;
	maskpad(1:N,1:M) = mask;

% Algorithm general metadata.
	if exist('mcalabmeta.mat','file'),
		load mcalabmeta;
		options = mcalabmeta;
	else
		disp('The original MCALab 110 metadata object has not been found.');
		disp('It should have been created in MCALABPATH/Utils/ subdirectory when launching MCAPath.');
		disp('Some meta information will not be saved in the options object.');
	end
	options.algorithm = 'ECM for 2D images';
	options.inpaint   = 'Inpainting: Yes';
	options.epsilon   = sprintf('Convergence tolerance: %e',epsilon);
	if ~exist('display','var'), % Default: no display.	
		display = 0; 		
		options.verbose = 'Verbose: No';	
	else
		options.verbose = 'Verbose: Yes';
	end
	[n,J]	= quadlength(imgpad);
	m0	= length(find(~maskpad(:)));
	options.thdtype = ['Threshold type: ' thdtype];

% Dictionary metadata.
	numberofdicts = LengthList(dict);
	options.nbcomp= sprintf('Number of morphological components: %d',numberofdicts);
	options.dict  = ['Transforms: [ ' dict ']'];
	str = 'Parameter 1 of transforms: [ ';
	for nb=1:numberofdicts, str = [str num2str(NthList(pars1,nb)) ' : ']; end
	options.pars1 = [str ']'];
	str = 'Parameter 2 of transforms: [ ';
	for nb=1:numberofdicts, str = [str num2str(NthList(pars2,nb)) ' : ']; end
	options.pars2 = [str ']'];	
	str = 'Parameter 3 of transforms: [ ';
	for nb=1:numberofdicts, str = [str num2str(NthList(pars3,nb)) ' : ']; end
	options.pars3 = [str ']'];
	part    = zeros(n,n,numberofdicts);
	imginp  = imgpad;
	
% To estimate the WGN standard deviation using the MAD.
% The qmf is quite arbitrary, sufficiently regular to approach a good band-pass.
if ~sigma | ~exist('sigma','var') | isempty(sigma)
	qmf=MakeONFilter('Daubechies',4);
	wc = FWT2_PO(imgpad,J-1,qmf);
	hh = wc(n/2+1:n/2+floor(N/2),n/2+1:n/2+floor(M/2));hh = hh(:);
	tmp   = maskpad(n/2+1:n/2+floor(N/2),n/2+1:n/2+floor(M/2));tmp = tmp(:);
	sigma = MAD(hh(find(tmp)));
	options.sigma = sprintf('Initial sigma estimated from data: %f',sigma);
else
	options.sigma = sprintf('Initial sigma fixed by the user: %f',sigma);
end
		
% Calculate the starting thd, which is the maximum of maximal coefficients 
% of the image in each dictionary, if lambda is decreased.
	switch ecmtype
		case 0
		   coeff = FastLA2(imgpad,dict,pars1,pars2,pars3); % In this case, this is only needed to normalize 
		   deltamax=StartingPoint(coeff,dict);		% the curvelet basis elements to unitary L2 norm.
		   delta=lambda*ones(itermax,1);
		   options.lambda=sprintf('Regularization parameter: lambda=%f',lambda);
		   itermax=1;
		   algmsg=sprintf('ECM inpainting with no std update');
		case 1
		   coeff = FastLA2(imgpad,dict,pars1,pars2,pars3);
	  	   deltamax=StartingPoint(coeff,dict);
		   delta=lambda*ones(itermax,1);
		   options.lambda=sprintf('Regularization parameter: lambda=%f',lambda);
		   T=(deltamax/(lambda*sigma)).^(2-2*[0:itermax-1]'/(itermax-1)); % Exponential cooling schedule.
		   sigma=sqrt(T(1))*sigma; % For the moment T is not used, but we initialize sigma at a high value.
		   itermax=1;
		   algmsg=sprintf('ECM inpainting with std update');
		case 2
		   coeff = FastLA2(imgpad,dict,pars1,pars2,pars3);
	  	   deltamax=StartingPoint(coeff,dict);
	  	   delta=-[0:itermax-1]'*(deltamax-lambda*sigma)/(itermax-1) + deltamax;	 % Linear decrease. 
	 	   delta=delta/sigma;
		   options.lambda      = lambda;
		   options.lambdamax   = sprintf('Starting threshold: %f',deltamax);
		   options.lambdasched = sprintf('Linear decrease schedule of threshold: step=%f',(deltamax-lambda*sigma)/(itermax-1));
		   algmsg=sprintf('Deterministic Annealing ECM inpainting');
		otherwise
		   error('Unknown EM inpainting algorithm specified.');
		   return;
	end
	options.algorithm = [algmsg ' for 2D images'];
	
	if display,
	  % Creates and returns a handle on the waitbar.
	  h = waitbar(0,sprintf('%s: Please wait...',algmsg));
	  figure(1);
	  clf
	  subplot(221);imagesc(imgpad);axis image;rmaxis;drawnow;
	  subplot(222);imagesc(imginp);axis image;rmaxis;drawnow;
	  % Save in an AVI movie file.
	  %aviobj = avifile(moviename);
	  %frame = getframe(gcf);
          %aviobj = addframe(aviobj,frame);
	  %clear frame
	end

% Start the ECM Algorithm.
for iter=0:itermax-1
t=1;imginpold=Inf;
while mean((imginpold(:)-imginp(:)).^2)>=epsilon & t<=1000
	  imginpold=imginp;
	  % E step: estimate the missing observations: 
	  %	Y^(t)_i =   Yobs_i 		if observation i is available 
	  %		 or Ymiss_i  		if observation is missing; Ymiss(t)=(1-M) Phi coeff^(t)
	    %yt=imgpad+(1-maskpad).*part;
	    %yt=imgpad-maskpad.*sum(part,3);
	  
	  % M step: solve for part the penalised least-squares problem conditionally on Y^(t), Phi and sigma^2(t)
	  %			min lambda * ||coeff||_1 +  ||img - M \Phi coeff||_2^2/(2 sigma(t)^2) -> Soft thresholding.
	    for nb=1:numberofdicts
	   % Multi-cycle EC step: estimate the missing observations: 
 	   %     Y^(t)_i =   Yobs_i	       if observation i is available 
 	   %		or Ymiss_i	       if observation is missing; Ymiss(t)=(1-M) Phi coeff^(t)
	      yt     = imgpad-maskpad.*imginp;
	      rt     = part(:,:,nb) + yt;
	      NAME   = NthList(dict,nb);
	      PAR1   = NthList(pars1,nb);
	      PAR2   = NthList(pars2,nb);
	      PAR3   = NthList(pars3,nb);
	      coeff  = FastLA2(rt,NAME,PAR1,PAR2,PAR3);
	      coeff  = eval([thdtype 'ThreshStruct(coeff,delta(iter+1)*sigma,NAME);']);
	      part(:,:,nb)   = FastLS2(coeff,NAME,PAR1,PAR2,PAR3);
	    end
	    
	  % M step: update the standard deviation
	    imginp  = sum(part,3);
	    if ecmtype==1
	    	sigma  = sqrt(m0*sigma^2 + sum(maskpad(:).*(imgpad(:)-imginp(:)).^2))/n;
	    end
	    %coeff = FastLA2(parts,dict,pars1,pars2,pars3);
	    %pll(iter+1)=norm(imgpad(:)-maskpad(:).*imginp(:))^2/2 + delta(iter+1)*sigma*normLp(coeff,1,dict);
	    s(iter+1)=sigma;
	    t=t+1;
	    if display,
	    	% Displays the progress time.
	    	figure(1);
	    	subplot(221);imagesc(imgpad);axis image;rmaxis;drawnow;
	    	subplot(222);imagesc(imginp);axis image;rmaxis;drawnow;
	    	%subplot(223);plot(pll);xlabel('Iteration');ylabel('PLL');
	    	%subplot(224);plot(s);xlabel('Iteration');ylabel('\sigma');
	    end
end	    
	if display,
	  % Displays the progress time.
	    waitbar((iter+1)/itermax,h);
	  % Save in an AVI movie file.
	    %if ~mod(iter,10)
	    %  frame = getframe(gcf);
            %  aviobj = addframe(aviobj,frame);
	    %  clear frame
	    %end
	end
end

	if display,
	  % Close the waitbar window
	  close(h);
	  %aviobj = close(aviobj);
	end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = ScadThreshStruct(C,lambda) 
% SCAD thresholding scheme of Fan et al. 99.

nbdicts = length(C);

for nb=1:nbdicts
 coeffs = C{nb};
 scaleindex = length(coeffs);

 for j = 2:scaleindex
 	  x=coeffs(j).coeff;
 	  tau=ones(size(x))*lambda;
  	  coeffs(j).coeff = sign(x).*max(abs(x)-tau,0).*(abs(x)<= 2*tau)...
			    + (abs(x) <= 3.7*tau) .* (abs(x) > 2*tau) .* (2.7*x-3.7.*tau.*sign(x))./1.7 ...
			    + (abs(x) > 3.7*tau).*x;
 end
 C{nb} = coeffs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = HardThreshStruct(C,lambda,nameofdict)
global E

nbdicts = length(C);

for nb=1:nbdicts
 coeffs = C{nb};
 scaleindex = length(coeffs);
 if strcmp(nameofdict,'CURVWRAP')
   for j = 2:scaleindex
     for w = 1:length(coeffs(j).coeff)
       coeffs(j).coeff{w} = coeffs(j).coeff{w}.* (abs(coeffs(j).coeff{w}) > lambda*E{j}{w});
     end
   end
 else
   for j = 2:scaleindex
  	  coeffs(j).coeff = coeffs(j).coeff .* (abs(coeffs(j).coeff) > lambda);
   end
 end
 C{nb} = coeffs;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = SoftThreshStruct(C,lambda,nameofdict)
global E

nbdicts = length(C);

for nb=1:nbdicts
 coeffs = C{nb};
 scaleindex = length(coeffs);
 if strcmp(nameofdict,'CURVWRAP')
   for j = 2:scaleindex
     for w = 1:length(coeffs(j).coeff)
       coeffs(j).coeff{w} = sign(coeffs(j).coeff{w}) .* max(abs(coeffs(j).coeff{w}) - lambda*E{j}{w},0);
     end
   end
 else
   for j = 2:scaleindex
  	  coeffs(j).coeff = sign(coeffs(j).coeff) .* max(abs(coeffs(j).coeff) - lambda,0);
   end
 end
 C{nb} = coeffs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta = StartingPoint(C,dict)
global E

nbdicts = length(C);

for nb=1:nbdicts
 tmp = [];
 coeffs = C{nb};
 scaleindex = length(coeffs);
 
 % If it is curvelet basis by the wrapping algorithm, then compute the L2 norm of the basis elements
 if strcmp(NthList(dict,nb),'CURVWRAP')
   computeL2norm(coeffs);
   for j = 2:scaleindex
     for w = 1:length(coeffs(j).coeff)
       wedge = coeffs(j).coeff{w}/E{j}{w};
       tmp = [tmp;wedge(:)];
     end
   end
 else
   for j = 2:scaleindex
  	  tmp = [tmp;coeffs(j).coeff(:)];
   end
 end
 buf(nb)=max(abs(tmp(:)));
end

%
buf=flipud(sort(buf(:),1));
if nbdicts>1 delta=buf(2);
else	     delta=buf(1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = TVCorrection(x,gamma)
% Total variation implemented using the approximate (exact in 1D) equivalence between the TV norm and the l_1 norm of the Haar (heaviside) coefficients.

[n,J] = quadlength(x);

qmf = MakeONFilter('Haar');

[ll,wc,L] = mrdwt(x,qmf,1);

wc = SoftThresh(wc,gamma);

y = mirdwt(ll,wc,qmf,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function computeL2norm(coeffs)
% Compute norm of curvelets (exact)
global E

%F = ones(size(coeffs(end).coeff{1}));
F = ones(coeffs(1).coeff{2});
X = fftshift(ifft2(F)) * sqrt(prod(size(F))); 	% Appropriately normalized Dirac
C = fdct_wrapping(X,1,length(coeffs)); 		% Get the curvelets

E = cell(size(C));
for j=1:length(C)
  E{j} = cell(size(C{j}));
  for w=1:length(C{j})
    A = C{j}{w};
    E{j}{w} = sqrt(sum(sum(A.*conj(A))) / prod(size(A)));
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lp = normLp(C,p,dict)
global E

nbdicts = length(C);
tmp = [];

for nb=1:nbdicts
 coeffs = C{nb};
 scaleindex = length(coeffs);
 
 if strcmp(NthList(dict,nb),'CURVWRAP')
   for j = 2:scaleindex
     for w = 1:length(coeffs(j).coeff)
       wedge = coeffs(j).coeff{w}/E{j}{w};
       tmp = [tmp;wedge(:)];
     end
   end
 else
   for j = 2:scaleindex
  	  tmp = [tmp;coeffs(j).coeff(:)];
   end
 end
end

%

lp=norm(tmp,p);
