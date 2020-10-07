function parts = firefrommetadata(x,mask,options)
% Script that reproduces results from metadata stored in options.
%
% Inputs:
%    x	     		1D signal or 2D image.
%    mask		The binary mask to be inpainted (signal or image of ones and zeros). An empty signal or an signal of ones  => no inpainting.
%    options		Structure containing all options and parameters of the algorithm, including default choices.
%  Outputs:
%    parti 		Estimated ith semantic component (nxn image)
%
%  Examples
%    % 1D MCA
%    x = GenSignal('EegfMRI');
%    load figMCAEEGmetadata
%    parts = firefrommetadata(x,[],options);
%
%    % 2D MCA
%    load linesgaussians
%    load figMCAlinesgaussiansmetadata
%    parts = firefrommetadata(img,[],options);
%
%    % 2D MCA inpainting.
%    load texturegaussians
%    mask=double(imread('mask_texturegaussians.bmp'));mask=mask(:,:,1);mask(mask~=0)=1;
%    load figInpainttexturegaussiansMCAmetadata
%    parts = firefrommetadata(img.*mask,mask,optionsMCA);
%
%   % 2D ECM inpainting.
%   load linesgaussians
%   mask=double(imread('mask_texturegaussians.bmp'));mask=mask(:,:,1);mask(mask~=0)=1;
%   load figInpaintlinesgaussiansECMmetadata
%   parts = firefrommetadata(img.*mask,mask,optionsECM);
%


if strcmp(options.algorithm,'MCA for 2D images') | strcmp(options.algorithm,'MCA for 1D signals')
	itermax = sscanf(options.itermax,'Number of iterations: %d');
	if isempty(str2num(options.tvcomponent(regexp(options.tvcomponent,'\d')))),
		tvcomponent = 0;
		tvregparam  = 0;
	else
		tvcomponent = str2num(options.tvcomponent(regexp(options.tvcomponent,'\d')));
		tvregparam  = str2num(options.tvregparam(regexp(options.tvregparam,'\d')));
	end
	
	if ~isempty(strfind(upper(options.inpaint),'YES')) & isempty(mask),
		error('Mask required for inpainting: see your metadata.');
	end
	
	if isempty(strfind(upper(options.verbose),'YES')),
		display = 0;
	else
		display = 1;
	end

	if isempty(strfind(upper(options.verbose),'LINEAR')),
		expdecrease = 0;
	else
		expdecrease = 1;
	end
	
	lambdastop = sscanf(options.stopcriterion,'Stopping threshold: %f x sigma=%f');lambdastop = lambdastop(1);
	sigma      = str2num(options.sigma(regexp(options.sigma,'[0-9.\-]')));
	
	% Dictionary metadata.
	numberofdicts = sscanf(options.nbcomp,'Number of morphological components: %d');
	spaces= strfind(options.dict,' ');
	dicts = options.dict(spaces(2)+1:spaces(3)-1);
	for nb=2:numberofdicts,
		dict  = options.dict(spaces(nb+1)+1:spaces(nb+2)-1);
		dicts = MakeList(dicts,dict);
	end
	spaces= strfind(options.pars1,':');
	if strfind(dicts,'LDCT2iv') | strfind(dicts,'LDCTiv') | strfind(dicts,'ALDCTiv'),
		pars1 = options.pars1(spaces(1)+4:spaces(2)-2);
		for nb=2:numberofdicts,
			pars1 = MakeList(pars1,options.pars1(spaces(nb)+2:spaces(nb+1)-2));
		end
	else
		pars1 = str2num(options.pars1(spaces(1)+3:spaces(2)-1));
		for nb=2:numberofdicts,
			pars1 = MakeList(pars1,str2num(options.pars1(spaces(nb)+2:spaces(nb+1)-1)));
		end
	end
	
	spaces= strfind(options.pars2,':');
	pars2 = str2num(options.pars2(spaces(1)+3:spaces(2)-1));
	for nb=2:numberofdicts,
		pars2 = MakeList(pars2,str2num(options.pars2(spaces(nb)+2:spaces(nb+1)-1)));
	end
	
	spaces= strfind(options.pars3,':');
	pars3 = str2num(options.pars3(spaces(1)+3:spaces(2)-1));
	for nb=2:numberofdicts,
		pars3 = MakeList(pars3,str2num(options.pars3(spaces(nb)+2:spaces(nb+1)-1)));
	end
	
	if strfind(options.algorithm,'1D')
		[parts,options]=MCA_Bcr(x,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,sigma,display);
	else
		[parts,options]=MCA2_Bcr(x,dicts,pars1,pars2,pars3,itermax,tvregparam,tvcomponent,expdecrease,lambdastop,mask,sigma,display);
	end
elseif strfind(upper(options.algorithm),'ECM')
	if isempty(mask),
		error('Mask required for inpainting: see your metadata.');
	end
	
	if isempty(strfind(upper(options.verbose),'YES')),
		display = 0;
	else
		display = 1;
	end
	
	epsilon   = sscanf(options.epsilon,'Convergence tolerance: %e');
	if strfind(options.algorithm,'ECM inpainting with no std update'),
		ecmtype = 0;
		lambda  = str2num(options.lambda(regexp(options.lambda,'[0-9.\-]')));
	elseif strfind(options.algorithm,'ECM inpainting with std update')
		ecmtype = 1;
		lambda  = str2num(options.lambda(regexp(options.lambda,'[0-9.\-]')));
	else
		ecmtype = 2;
		lambda  = options.lambda;
	end	
		
	sigma = str2num(options.sigma(regexp(options.sigma,'[0-9.\-]')));
	
	thdtype = sscanf(options.thdtype,'Threshold type: %s');
		
	% Dictionary metadata.
	numberofdicts = sscanf(options.nbcomp,'Number of morphological components: %d');
	spaces= strfind(options.dict,' ');
	dicts = options.dict(spaces(2)+1:spaces(3)-1);
	for nb=2:numberofdicts,
		dict  = options.dict(spaces(nb+1)+1:spaces(nb+2)-1);
		dicts = MakeList(dicts,dict);
	end
	spaces= strfind(options.pars1,':');
	if strfind(dicts,'LDCT2iv') | strfind(dicts,'LDCTiv') | strfind(dicts,'ALDCTiv'),
		pars1 = options.pars1(spaces(1)+3:spaces(2)-1);
		for nb=2:numberofdicts,
			pars1 = MakeList(pars1,options.pars1(spaces(nb)+2:spaces(nb+1)-1));
		end
	else
		pars1 = str2num(options.pars1(spaces(1)+3:spaces(2)-1));
		for nb=2:numberofdicts,
			pars1 = MakeList(pars1,str2num(options.pars1(spaces(nb)+2:spaces(nb+1)-1)));
		end
	end
	
	spaces= strfind(options.pars2,':');
	pars2 = str2num(options.pars2(spaces(1)+3:spaces(2)-1));
	for nb=2:numberofdicts,
		pars2 = MakeList(pars2,str2num(options.pars2(spaces(nb)+2:spaces(nb+1)-1)));
	end
	
	spaces= strfind(options.pars3,':');
	pars3 = str2num(options.pars3(spaces(1)+3:spaces(2)-1));
	for nb=2:numberofdicts,
		pars3 = MakeList(pars3,str2num(options.pars3(spaces(nb)+2:spaces(nb+1)-1)));
	end

	parts = EM_Inpaint(x,dicts,pars1,pars2,pars3,1,epsilon,lambda,sigma,thdtype,ecmtype,mask,display);

else
	disp('Uknown algorithm. Your metadata may be corrupted.');
end
	

