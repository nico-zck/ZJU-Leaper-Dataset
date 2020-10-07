function [part, options] = MCA2_Bcr(img, dict, pars1, pars2, pars3, itermax, gamma, comptv, expdecrease, stop, mask, sigma, display)
% MCA2_Bcr: Morphological Component Analysis of a 2D images (a matrix) using highly redundant dictionaries and sparstity promoting penalties.
%	   The optimization pb is solved using a modified version of the BCR algorithm.
%	   MCA2_bcr solves the following optimization problem
%		(part_i,for all i) = argmin Sum_i || \Phi^+_i part_i ||_p + lambda * ||img - Sum_i part_i||_2^2
%			Sparsity is promoted by the lp norm. Ideally the l_0 norm but this is relaxed to the l_1 norm.
%				p = 1 (l_1 norm: Soft thesholding as a solution).
%				p = 0 (l_0 norm: difficult but approximated with a Hard thresholding).
%	   Each component is supposed to be sparsely described in its corresponding dictionary \Phi.
%  Usage:
%    part=MCA2_Bcr(img,dict,pars1,pars2,pars3,itermax,gamma,expdecrease,stop,mask)
%  Inputs:
%    img	 	2D image matrix, nxn, if n ~= 2^J, use zero-padding and crop at the end.
%    dict		Names of dictionaries for each part (see directory Dictionary for available transforms)
%    parsi		Parameters of dictionary (built using the MakeList function)
%    itermax	Nb of relaxation iterations
%    gamma   	TV regularization parameter (usually applied to the piece-wise smooth component, e.g. UDWT or curvelet dictionary).
%    comptv  	Component to which TV regularization is applied.
%    expdecrease	Exponential/Linear decrease of the regularization parameter
%    stop	 	Stop criterion, the algorithm stops when lambda <= stop*sigma (typically k=3), sigma is the noise WGN std
%    mask		The binary mask to be inpainted (image of ones and zeros). An image of ones => no inpainting.
%    sigma		Value of noise std. If not provided, it will be estimated (default).
%    display	Display algorithm progress [0-none, 1-bar, 2-bar and figure].
%  Outputs:
%    parti 		Estimated ith semantic component (nxn image)
%    options	Structure containing all options and parameters of the algorithm, including default choices.
%
%  Description
%    The dictionaries and their parameters can be built using the MakeList function.
%    A demo GUI (MCADemo) can be called to guide the user in these steps.
%  See Also
%    FastLA2, FastLS2, MCA2Demo

global E% Energy (L2 norm) of curvelets if CURVWRAP is used in the dictionary

% Initializations.
[N, M] = size(img);
n = 2^(nextpow2(max(N, M)));
imgpad = zeros(n, n);
maskpad = ones(n, n);
imgpad(1:N, 1:M) = img;

% Algorithm general metadata.
if exist('mcalabmeta.mat', 'file'),
    load mcalabmeta;
    options = mcalabmeta;
else
    disp('The original MCALab 110 metadata object has not been found.');
    disp('It should have been created in MCALABPATH/Utils/ subdirectory when launching MCAPath.');
    disp('Some meta information will not be saved in the options object.');
end

options.algorithm = 'MCA for 2D images';
options.itermax = sprintf('Number of iterations: %d', itermax);

if comptv & gamma,
    options.tvregparam = sprintf('TV regularization parameter: %f', gamma);
    options.tvcomponent = sprintf('TV-regularized component: %d', comptv);
else
    options.tvregparam = 'TV regularization parameter: None';
    options.tvcomponent = 'TV-regularized component: None';
end

if exist('mask', 'var') & ~isempty(mask),
    maskpad(1:N, 1:M) = mask;
    options.inpaint = 'Inpainting: Yes';
else
    options.inpaint = 'Inpainting: No';
end;

if ~exist('display', 'var'), % Default: no display.
    display = 0;
    options.verbose = 'Verbose: No';
else
    options.verbose = 'Verbose: Yes';
end

[n, J] = quadlength(imgpad);
thdtype = 'Hard';
options.thdtype = ['Threshold type: ' thdtype];

% Dictionary metadata.
numberofdicts = LengthList(dict);
options.nbcomp = sprintf('Number of morphological components: %d', numberofdicts);
options.dict = ['Transforms: [ ' dict ']'];
str = 'Parameter 1 of transforms: [ ';
for nb = 1:numberofdicts, str = [str num2str(NthList(pars1, nb)) ' : ']; end
options.pars1 = [str ']'];
str = 'Parameter 2 of transforms: [ ';
for nb = 1:numberofdicts, str = [str num2str(NthList(pars2, nb)) ' : ']; end
options.pars2 = [str ']'];
str = 'Parameter 3 of transforms: [ ';
for nb = 1:numberofdicts, str = [str num2str(NthList(pars3, nb)) ' : ']; end
options.pars3 = [str ']'];
part = zeros(n, n, numberofdicts);

% To estimate the WGN standard deviation using the MAD.
% The qmf is quite arbitrary, sufficiently regular to approach a good band-pass.
if ~exist('sigma', 'var') | isempty(sigma),
    qmf = MakeONFilter('Daubechies', 4);
    wc = FWT2_PO(imgpad, J - 1, qmf);
    hh = wc(n / 2 + 1:n / 2 + floor(N / 2), n / 2 + 1:n / 2 + floor(M / 2)); hh = hh(:);
    tmp = maskpad(n / 2 + 1:n / 2 + floor(N / 2), n / 2 + 1:n / 2 + floor(M / 2)); tmp = tmp(:);
    sigma = MAD(hh(find(tmp)));
    options.sigma = sprintf('Initial sigma estimated from data: %f', sigma);
else
    options.sigma = sprintf('Initial sigma fixed by the user: %f', sigma);
end

stopcriterion = stop * sigma;
options.stopcriterion = sprintf('Stopping threshold: %f x sigma=%f', stop, stopcriterion);

% First pass: coeffs of the original image in each dictionary.
coeff = FastLA2(imgpad, dict, pars1, pars2, pars3);

% Calculate the starting thd, which is the minimum of maximal coefficients
% of the image in each dictionary.
deltamax = StartingPoint(coeff, dict);
delta = deltamax;
options.lambdamax = sprintf('Starting threshold: %f', deltamax);

if expdecrease
    lambda = (deltamax / stopcriterion)^(1 / (1 - itermax)); % Exponential decrease.
    options.lambdasched = sprintf('Exponential decrease schedule of threshold: step=%f', lambda);
else
    lambda = (deltamax - stopcriterion) / (itermax - 1); % Slope of the linear decrease.
    options.lambdasched = sprintf('Linear decrease schedule of threshold: step=%f', lambda);
end

if display > 0
    % Create and return a handle on the waitbar.
    handle = waitbar(0, 'MCA in progress: Please wait...');
    nbpr = ceil(sqrt(numberofdicts + 2));
    if display > 1
        figure(1); clf
        subplot(nbpr, nbpr, 1); imagesc(imgpad(1:N, 1:M)); axis image; rmaxis; drawnow;
        subplot(nbpr, nbpr, 2); imagesc(sum(part(1:N, 1:M, :), 3)); axis image; rmaxis; title('\Sigma_i Part_i'); drawnow;
        for np = 1:numberofdicts
            subplot(nbpr, nbpr, np + 2); imagesc(part(1:N, 1:M, np)); axis image; rmaxis; title(sprintf('Part_%d', np)); drawnow;
        end
    end
    
    % Save in an AVI movie file.
    %aviobj = avifile('barbara_MCA_inpaint.avi');
    %frame = getframe(gcf);
    %aviobj = addframe(aviobj,frame);
    %clear frame
end

% Start the modified Block Relaxation Algorithm.
for iter = 0:itermax - 1
    %for i=1:J
    % Calculate the residual image.
    residual = imgpad - maskpad .* sum(part, 3);
    
    % Cycle over dictionaries.
    for nb = 1:numberofdicts
        % Update Parta assuming other parts fixed.
        % Solve for Parta the marginal penalized minimization problem (Hard thesholding, l_1 -> Soft).
        NAME = NthList(dict, nb);
        PAR1 = NthList(pars1, nb);
        PAR2 = NthList(pars2, nb);
        PAR3 = NthList(pars3, nb);
        Ra = part(:, :, nb) + residual;
        coeffa = FastLA2(Ra, NAME, PAR1, PAR2, PAR3);
        coeffa = eval([thdtype 'ThreshStruct(coeffa,delta,NAME);']);
        part(:, :, nb) = FastLS2(coeffa, NAME, PAR1, PAR2, PAR3);
        if (nb == comptv) & gamma ~= 0, part(:, :, nb) = TVCorrection(part(:, :, nb), gamma); end
    end
    
    %end
    
    % Update the regularization parameter delta.
    if expdecrease	delta = delta * lambda; % Exponential decrease.
    else delta = delta - lambda; % Linear decrease.
    end
    
    % Displays the progress.
    if display > 0
        waitbar((iter + 1) / itermax, handle, [num2str((iter + 1)), '/', num2str(itermax)]);
        if display > 1
            figure(1);
            subplot(nbpr, nbpr, 1); imagesc(imgpad(1:N, 1:M)); axis image; rmaxis; drawnow;
            subplot(nbpr, nbpr, 2); imagesc(sum(part(1:N, 1:M, :), 3)); axis image; rmaxis; title('\Sigma_i Part_i'); drawnow;
            for np = 1:numberofdicts
                subplot(nbpr, nbpr, np + 2); imagesc(part(1:N, 1:M, np)); axis image; rmaxis; title(sprintf('Part_%d', np)); drawnow;
            end
        end
        
        % Save in an AVI movie file.
        %frame = getframe(gcf);
        %aviobj = addframe(aviobj,frame);
        %clear frame
    end
    
end

if display > 0
    % Close the waitbar window
    close(handle);
    if display > 1
        figure(1);
        subplot(nbpr, nbpr, 1); imagesc(imgpad(1:N, 1:M)); axis image; rmaxis; drawnow;
        subplot(nbpr, nbpr, 2); imagesc(sum(part(1:N, 1:M, :), 3)); axis image; rmaxis; title('\Sigma_i Part_i'); drawnow;
        for np = 1:numberofdicts
            subplot(nbpr, nbpr, np + 2); imagesc(part(1:N, 1:M, np)); axis image; rmaxis; title(sprintf('Part_%d', np)); drawnow;
        end
    end
    % Close the AVI movie flow.
    %aviobj = close(aviobj);
end

% Crop data to original size.
part = part(1:N, 1:M, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = HardThreshStruct(C, lambda, nameofdict)
global E

nbdicts = length(C);

for nb = 1:nbdicts
    coeffs = C{nb};
    scaleindex = length(coeffs);
    
    if strcmp(nameofdict, 'CURVWRAP')
        
        for j = 2:scaleindex
            
            for w = 1:length(coeffs(j).coeff)
                coeffs(j).coeff{w} = coeffs(j).coeff{w} .* (abs(coeffs(j).coeff{w}) > lambda * E{j}{w});
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
function C = SoftThreshStruct(C, lambda, nameofdict)
global E

nbdicts = length(C);

for nb = 1:nbdicts
    coeffs = C{nb};
    scaleindex = length(coeffs);
    
    if strcmp(nameofdict, 'CURVWRAP')
        
        for j = 2:scaleindex
            
            for w = 1:length(coeffs(j).coeff)
                coeffs(j).coeff{w} = sign(coeffs(j).coeff{w}) .* max(abs(coeffs(j).coeff{w}) - lambda * E{j}{w}, 0);
            end
            
        end
        
    else
        
        for j = 2:scaleindex
            coeffs(j).coeff = sign(coeffs(j).coeff) .* max(abs(coeffs(j).coeff) - lambda, 0);
        end
        
    end
    
    C{nb} = coeffs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta = StartingPoint(C, dict)
global E

nbdicts = length(C);

for nb = 1:nbdicts
    tmp = [];
    coeffs = C{nb};
    scaleindex = length(coeffs);
    
    % If it is curvelet basis by the wrapping algorithm, then compute the L2 norm of the basis elements
    if strcmp(NthList(dict, nb), 'CURVWRAP')
        computeL2norm(coeffs);
        
        for j = 2:scaleindex
            
            for w = 1:length(coeffs(j).coeff)
                wedge = coeffs(j).coeff{w} / E{j}{w};
                tmp = [tmp; wedge(:)];
            end
            
        end
        
    else
        
        for j = 2:scaleindex
            tmp = [tmp; coeffs(j).coeff(:)];
        end
        
    end
    
    buf(nb) = max(abs(tmp(:)));
end

%
buf = flipud(sort(buf(:), 1))';

if nbdicts > 1 delta = buf(2);
else delta = buf(1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = TVCorrection(x, gamma)
% Total variation implemented using the approximate (exact in 1D) equivalence between the TV norm and the l_1 norm of the Haar (heaviside) coefficients.

[n, J] = quadlength(x);

qmf = MakeONFilter('Haar');

[ll, wc, L] = mrdwt(x, qmf, 1);

wc = SoftThresh(wc, gamma);

y = mirdwt(ll, wc, qmf, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function computeL2norm(coeffs)
% Compute norm of curvelets (exact)
global E

%F = ones(size(coeffs(end).coeff{1}));
F = ones(coeffs(1).coeff{2});
X = fftshift(ifft2(F)) * sqrt(prod(size(F))); % Appropriately normalized Dirac
C = fdct_wrapping(X, 1, length(coeffs)); % Get the curvelets

E = cell(size(C));

for j = 1:length(C)
    E{j} = cell(size(C{j}));
    
    for w = 1:length(C{j})
        A = C{j}{w};
        E{j}{w} = sqrt(sum(sum(A .* conj(A))) / prod(size(A)));
    end
    
end
