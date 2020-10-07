function F = MakeDict(n, NameOfDict, par1, par2, par3)
% MakeDict -- Build the basis matrix \Phi for a dictionary
% Usage:
%	F = MakeDict(n, NameOfDict, par1, par2, par3)
% Input:
%	n		the sample size
%	NameOfDict	name of the dictionary
%	par1,par2,par3	the parameters of the dictionary
%
%	Use 'help dictionary' for dictionary objects: NameOfDict,par1,par2,par3
% Output:
%	F	the basis matrix, by column


if nargin < 5,
	par3 = 0;
end
if nargin < 4,
        par2 = 0;
end
if nargin < 3,
        par1 = 0;
end


[m L] = SizeOfDict(n, NameOfDict, par1, par2, par3);
zerosm = zeros(m,1);
F = zeros(n, m);
for i = 1:m,
	c = zerosm; c(i) = 1;
	F(:,i) = FastS(c, n, NameOfDict, par1, par2, par3);
end
