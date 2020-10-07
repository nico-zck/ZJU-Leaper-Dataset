function [m, L] = SizeOfDict1(n, NameOfDict, par1, par2, par3)
% SizeOfDict -- the size of a (merged) dictionary
%  Usage:
%	[m, L] = SizeOfDict2(n, NameOfDict, par1, par2, par3)
%  Inputs:
%	n		the image size (n x n)
%	NameOfDict	name of the dictionary
%	par1,par2,par3	the parameters of the dictionary
%
%	Use 'help dictionary' for dictionary objects: NameOfDict,par1,par2,par3
%  Outputs:
%	m		# of basis functions
%	L		# redundancy = m/n
% 

NumberOfDicts = LengthList(NameOfDict);
m = 0;
L = 0;
for i = 1:NumberOfDicts,
	NAME = NthList(NameOfDict, i);
        PAR1 = NthList(par1, i);
        PAR2 = NthList(par2, i);
        PAR3 = NthList(par3, i);

	if strcmp(NAME, 'DIRAC'),
		L0 = 1;
		m0 = n;
	elseif strcmp(NAME, 'PO'),
		L0 = 1;
		m0 = n;
	elseif strcmp(NAME, 'UDWT'),
		J = nextpow2(n);
		L0 = (J-PAR1) + 1;
		m0 = L0*n;
	elseif strcmp(NAME, 'UDWTTIGHT'),
		J = nextpow2(n);
		L0 = (J-PAR1) + 1;
		m0 = L0*n;
	elseif strcmp(NAME, 'PBS'),
		L0 = 1;
		m0 = n;
	elseif strcmp(NAME, 'POSpin'),
		L0 = 2*PAR3+1;
		m0 = L0*n;
	elseif strcmp(NAME, 'ATrou'),
		J = nextpow2(n);
		L0 = (J-PAR1) + 1;
		m0 = L0*n;
	elseif strcmp(NAME, 'WP'),
		L0 = PAR1 + 1;
		m0 = L0*n;
	elseif strcmp(NAME, 'DCT'),
		L0 = PAR1;
		m0 = L0*n;
	elseif strcmp(NAME, 'DST'),
		L0 = PAR1;
		m0 = L0*n;
	elseif strcmp(NAME, 'DSTi'),
		L0 = PAR1;
		m0 = L0*n;
	elseif strcmp(NAME, 'LDCT'),
		L0 = 2*PAR2+1;
		m0 = L0*n;
	elseif strcmp(NAME, 'LDCTiv'),
		L0 = 1;
		m0 = L0*n;
	elseif strcmp(NAME, 'CP'),
		L0 = PAR1 + 1;
		m0 = L0*n;
	elseif strcmp(NAME, 'RealFourier'),
		L0 = 2;
		m0 = L0*n;
	end

	m = m + m0; L = L + L0;
end
