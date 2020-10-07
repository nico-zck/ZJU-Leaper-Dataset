function [part,options]=MCA_Bcr(signal,dict,pars1,pars2,pars3,itermax,gamma,comptv,expdecrease,stop,mask,sigma,display)
% MCA_Bcr: Morphological Component Analysis of a 1D signal (a vector) using highly redundant dictionaries.
%	   The optimization pb is solved using a modified version of the BCR algorithm.
%	   MCA_bcr solves the following optimization problem
%		(part_i,for all i) = argmin Sum_i || \Phi^+_i part_i ||_p + lambda * ||img - Sum_i part_i||_2^2
%			Sparsity is promoted by the lp norm. Ideally the l_0 norm but this is relaxed to the l_1 norm.	
%				p = 1 (l_1 norm: Soft thesholding as a solution).
%				p = 0 (l_0 norm: difficult but approximated with a Hard thresholding).
%	   Each component is supposed to be sparsely described in its corresponding dictionary \Phi.
%  Usage:
%    part=MCA_Bcr(signal,dict,pars1,pars2,pars3,itermax,gamma,comptv,expdecrease,stop,mask)
%  Inputs:
%    signal     	1D signal vector, n = 2^J, if n ~= 2^J, use zero-padding and crop at the end.
%    dict		Names of dictionaries for each part (see directory Dictionary for available transforms)
%    parsi		Parameters of dictionary (built using the MakeList function)
%    itermax		Nb of relaxation iterations
%    gamma      	TV regularization parameter (usually applied to the piece-wise smooth component, e.g. UDWT or curvelet dictionary).
%    comptv      	Component to which TV regularization is applied.
%    expdecrease	Exponential/Linear decrease of the regularization parameter
%    stop	 	Stop criterion, the algorithm stops when lambda <= stop*sigma (typically k=3), sigma is the noise WGN std
%    mask		The binary mask to be inpainted (image of ones and zeros). An image of ones => no inpainting.
%    sigma		Value of noise std. If not provided, it will be estimated (default).
%    display		Display algorithm progress. 
%  Outputs:
%    parti 		Estimated ith semantic component (1D signal).
%    options		Structure containing all options and parameters of the algorithm, including default choices.
%
%  Description
%    The dictionaries and their parameters can be built using the MakeList function.
%    A demo GUI (MCADemo) can be called to guide the user in these steps. 
%  See Also
%    FastLA, FastLS, MCADemo


% Initializations. Put the signal as a column vector.
	signal	      = signal(:);
	N  	      = length(signal);
	n      	      = 2^(nextpow2(N));
	sigpad 	      = zeros(n,1);
	maskpad	      = ones(n,1);
	sigpad(1:N)   = signal;
	
% Algorithm general metadata.
	if exist('mcalabmeta.mat','file'),
		load mcalabmeta;
		options = mcalabmeta;
	else
		disp('The original MCALab 110 metadata object has not been found.');
		disp('It should have been created in MCALABPATH/Utils/ subdirectory when launching MCAPath.');
		disp('Some meta information will not be saved in the options object.');
	end
	options.algorithm = 'MCA for 1D signals';
	options.itermax   = sprintf('Number of iterations: %d',itermax);
	if comptv & gamma, 
		options.tvregparam     = sprintf('TV regularization parameter: %f',gamma);
		options.tvcomponent    = sprintf('TV-regularized component: %d',comptv);
	else	
		options.tvregparam     = 'TV regularization parameter: None';	
		options.tvcomponent    = 'TV-regularized component: None';
	end
	if exist('mask','var') & ~isempty(mask),  	
		maskpad(1:N) = mask;
		options.inpaint = 'Inpainting: Yes'; 
	else
		options.inpaint = 'Inpainting: No'; 
	end;
	if ~exist('display','var'), % Default: no display.	
		display = 0; 		
		options.verbose = 'Verbose: No';	
	else
		options.verbose = 'Verbose: Yes';
	end
	[n,J]	      = dyadlength(sigpad);
	thdtype       = 'Hard';
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
	part	      = zeros(n,numberofdicts);

% To estimate the WGN standard deviation using the MAD.
% The qmf is quite arbitrary, sufficiently regular to approach a good band-pass.
	if ~exist('sigma','var') | isempty(sigma),
	  qmf=MakeONFilter('Daubechies',4);
	  wc = FWT_PO(sigpad,J-1,qmf);
	  sigma = MAD(wc(n/2+1:n/2+floor(N/2)));
	  options.sigma = sprintf('Sigma estimated from data: %f',sigma);
	else
	  options.sigma = sprintf('Sigma fixed by the user: %f',sigma);
	end
	stopcriterion = stop*sigma;
	options.stopcriterion = sprintf('Stopping threshold: %f x sigma=%f',stop,stopcriterion);

% First pass: coeffs of the original signl in each dictionary.
	coeff = FastLA(sigpad,dict,pars1,pars2,pars3);

% Calculate the starting thd, which is the minimum of maximal coefficients 
% of the signal in each dictionary.
	deltamax=StartingPoint(coeff);
	delta=deltamax;
	options.lambdamax   = sprintf('Starting threshold: %f',deltamax);
	if expdecrease	
		lambda=(deltamax/stopcriterion)^(1/(1-itermax)); % Exponential decrease.
		options.lambdasched = sprintf('Exponential decrease schedule of threshold: step=%f',lambda);
	else	 	
		lambda=(deltamax-stopcriterion)/(itermax-1);	 % Slope of the linear decrease. 
		options.lambdasched = sprintf('Linear decrease schedule of threshold: step=%f',lambda);
	end
	
	if display,
	  % Create and return a handle on the waitbar.
	  h = waitbar(0,'MCA in progress: Please wait...');
	  nbpr=ceil(sqrt(numberofdicts+2));
	  figure(1);
	  subplot(nbpr,nbpr,1);plot(signal);axis tight;drawnow;
	  subplot(nbpr,nbpr,2);plot(sum(part(1:N,:),2));title('\Sigma_i Part_i');axis tight;drawnow;
	  for np=1:numberofdicts
	    subplot(nbpr,nbpr,np+2);plot(part(1:N,np));title(sprintf('Part_%d',np));axis tight;drawnow;
	  end
	end

% Start the modified Block Relaxation Algorithm.
for iter=0:itermax-1
	%for i=1:1
	  % Calculate the residual signal.
	    residual=sigpad-maskpad.*sum(part,2);
	    
	  % Cycle over dictionaries.
	   for nb=1:numberofdicts
	   % Update Parta assuming other parts fixed.
	   % Solve for Parta the marginal penalized minimization problem (Hard thesholding, l_1 -> Soft).
	     NAME   = NthList(dict,nb);
	     PAR1   = NthList(pars1,nb);
	     PAR2   = NthList(pars2,nb);
	     PAR3   = NthList(pars3,nb);
	     Ra=part(:,nb)+residual;
	     Ca = FastLA(Ra,NAME,PAR1,PAR2,PAR3);
	     coeffa = Ca{1};
	     coeffa(2).coeff = eval([thdtype 'Thresh(coeffa(2).coeff,delta)']); % Do not threshold low-frequency coefficients.
	     Ca{1} = coeffa;
	     part(:,nb)  = FastLS(Ca,NAME,PAR1,PAR2,PAR3);
	     if (nb == comptv) & gamma~=0, part(:,nb) = TVCorrection(part(:,nb),gamma); end
	   end
	%end
	
	% Update the regularization parameter delta.
	    if expdecrease	delta=delta*lambda; % Exponential decrease.	
	    else		delta=delta-lambda; % Linear decrease.
	    end
	
	% Displays the progress.
	    if display,
	      waitbar((iter+1)/itermax,h);	    
	      figure(1);
	      subplot(nbpr,nbpr,1);plot(signal);axis tight;drawnow;
	      subplot(nbpr,nbpr,2);plot(sum(part(1:N,:),2));title('\Sigma_i Part_i');axis tight;drawnow;
	      for np=1:numberofdicts
	    	subplot(nbpr,nbpr,np+2);plot(part(1:N,np));title(sprintf('Part_%d',np));axis tight;drawnow;
	      end
	      %M(iter+1)=getframe(gcf);
	    end
end
	

	if display,
	  % Closes the waitbar window
	  close(h);
	  figure(1);
	  subplot(nbpr,nbpr,1);plot(signal);axis tight;drawnow;
	  subplot(nbpr,nbpr,2);plot(sum(part(1:N,:),2));title('\Sigma_i Part_i');axis tight;drawnow;
	  for np=1:numberofdicts
	    subplot(nbpr,nbpr,np+2);plot(part(1:N,np));title(sprintf('Part_%d',np));axis tight;drawnow;
	  end
	  %movie2avi(M,'MCA1Demo.avi');
	end

% Crop data to original size.
	part = part(1:N,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta = StartingPoint(C)

nbdicts = length(C);

for nb=1:nbdicts
 tmp = [];
 coeffs = C{nb};
 scaleindex = length(coeffs);

 for j = 2:scaleindex
  	  tmp = [tmp;coeffs(j).coeff(:)];
 end
 buf(nb)=max(abs(tmp(:)));
end

%
buf=flipud(sort(buf(:),1))';
delta=buf(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
function y = TVCorrection(x,gamma)
% Total variation implemented using the equivalence between the TV norm and the l_1 norm of the Haar (heaviside) coefficients.

[n,J] = dyadlength(x);

qmf = MakeONFilter('Haar');

[ll,wc,L] = mrdwt(x,qmf,1);

wc = SoftThresh(wc,gamma);

y = mirdwt(ll,wc,qmf,1);

