%  MCALab110 -- initialize path to include MCALab110 
%
    global MCALAB WAVELABVER CURVELABVER RWTVER
    
% Current version of distributed library. Update to match your actual versions.
    MCALAB      = 110;
    WAVELABVER  = num2str(802);
    CURVELABVER = '2.1.1';
    RWTVER      = '1.2';
    BVstr=num2str(MCALAB);
    
	fprintf('\nWelcome to MCALab v %g\n\n', MCALAB);
%
	global MCALABPATH
	global PATHNAMESEPARATOR
	global MATLABPATHSEPARATOR
   	
%
	Friend = computer;
	if strcmp(Friend,'MAC2'),
	  PATHNAMESEPARATOR = ':';
	  MCALABPATH = ['Macintosh HD:Build:MCALabWithUtilities', BVstr, PATHNAMESEPARATOR];
	  MATLABPATHSEPARATOR = ';';
	elseif isunix,
	  PATHNAMESEPARATOR = '/';
	  [s,HOME]	    = unix('echo $HOME');
	  MCALABPATH = ['/Users/jalal/Matlab/WaveLab802/MCA802/downloads/MCALabWithUtilities/', PATHNAMESEPARATOR];
	  MATLABPATHSEPARATOR = ':';
	elseif strcmp(Friend(1:2),'PC');
	  PATHNAMESEPARATOR = '\';	  
% 	  MCALABPATH = [matlabroot,'\toolbox\MCALabWithUtilities', BVstr, PATHNAMESEPARATOR];  
      MCALABPATH = ['D:\Workspace\matlab\LatticeSeg\MCALabWithUtilities\']; 
	  MATLABPATHSEPARATOR = ';';
	else
		disp('I don''t recognize this computer; ')
		disp('Pathnames not set; solution: edit SparsePath.m\n\n')
	end
%
	global MATLABVERSION
	V = version;
	MATLABVERSION = str2num(V(1:3));

    if MATLABVERSION < 5.3,
        disp('Warning: This version is only supported on Matlab 6.x and higher');
        MCAtoolbox=genpath(MCALABPATH,1);
    else
        MCAtoolbox=genpath(MCALABPATH);
    end
    
    path(MCAtoolbox,path);  
     
%
	fprintf('Setting Global Variables:\n');
	fprintf('   global MATLABVERSION = %g\n',	MATLABVERSION)
	fprintf('   global MCALABPATH = %s\n',		MCALABPATH)
    
%
	fprintf('   MCALab100 Path set successfully.\n');
	

% Fill the metadata object with central information.
	mcalabmeta.matlabver   = ['Matlab version ' V];
	mcalabmeta.arch	       = ['Architecture ' Friend];
	mcalabmeta.mcalabver   = ['MCALab version ' BVstr];
	mcalabmeta.wavelabver  = ['WaveLab version ' WAVELABVER];
	mcalabmeta.curvrlabver = ['CurveLab version ' CURVELABVER];
	mcalabmeta.rwtver      = ['RWT version ' RWTVER];
	mcalabmetapathname     = [MCALABPATH '/Utils/mcalabmeta.mat'];
	eval(sprintf('save %s mcalabmeta',mcalabmetapathname));
	
	clear all 
	
% Check if WaveLab80X is installed. Just test if FWT_PO is in the path.
	if isempty(which('FWT_PO')),
		error('WaveLab does not seem to be installed, please check and run again MCAPath');
	end
	
	
